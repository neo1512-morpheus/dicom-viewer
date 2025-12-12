import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from "https://esm.sh/@supabase/supabase-js@2"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// Helper function to sign a single instance
async function signInstanceUrl(supabaseClient, patientId, instance) {
  if (instance.url && !instance.url.startsWith('http')) {
    const filePath = `${patientId}/${instance.url}`
    const { data: signedData } = await supabaseClient
      .storage
      .from('scans')
      .createSignedUrl(filePath, 3600) // 1 hour expiry

    if (signedData?.signedUrl) {
      // Mutate the instance object directly
      instance.url = `dicomweb:${signedData.signedUrl}`
    }
  }
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    const url = new URL(req.url)
    const patientId = url.searchParams.get('patientId')

    if (!patientId) throw new Error('Missing patientId')

    // --- CACHING LAYER ---
    // If we have a cached response for this URL, return it.
    // Supabase Edge Runtime handles this automatically with Cache-Control headers.

    console.log(`Fetching manifest for: ${patientId}`)

    const { data: manifestData, error: downloadError } = await supabaseClient
      .storage
      .from('scans')
      .download(`${patientId}/dicom_manifest.json`)

    if (downloadError) throw downloadError

    const manifestText = await manifestData.text()
    const manifest = JSON.parse(manifestText)

    // --- PARALLEL SIGNING ---
    const signingPromises = [];
    const studies = manifest.studies || []
    for (const study of studies) {
      const seriesList = study.series || []
      for (const series of seriesList) {
        const instances = series.instances || []
        for (const instance of instances) {
          // Add the signing task to our list
          signingPromises.push(signInstanceUrl(supabaseClient, patientId, instance));
        }
      }
    }

    // Wait for ALL simultaneous requests to finish. This is much faster.
    await Promise.all(signingPromises);

    return new Response(JSON.stringify(manifest), {
      headers: { 
        ...corsHeaders, 
        'Content-Type': 'application/json',
        // Cache this generated JSON for 50 minutes. 
        // Subsequent reloads within this time will be INSTANT.
        'Cache-Control': 'public, s-maxage=3000' 
      },
      status: 200,
    })

  } catch (error) {
    console.error('Function Error:', error)
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 400,
    })
  }
})