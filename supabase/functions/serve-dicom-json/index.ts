import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { S3Client, GetObjectCommand } from "npm:@aws-sdk/client-s3";
import { getSignedUrl } from "npm:@aws-sdk/s3-request-presigner";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// Initialize S3 Client for R2
const s3Client = new S3Client({
  region: 'auto',
  endpoint: `https://${Deno.env.get('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: Deno.env.get('R2_ACCESS_KEY_ID') ?? '',
    secretAccessKey: Deno.env.get('R2_SECRET_ACCESS_KEY') ?? ''
  }
});

const bucketName = Deno.env.get('R2_BUCKET_NAME') ?? '';

// Helper function to sign a single instance URL using R2/S3
async function signInstanceUrl(patientId: string, instance: any) {
  if (instance.url && !instance.url.startsWith('http')) {
    const filePath = `${patientId}/${instance.url}`;

    try {
      const command = new GetObjectCommand({
        Bucket: bucketName,
        Key: filePath
      });

      const signedUrl = await getSignedUrl(s3Client, command, { expiresIn: 3600 });

      // Mutate the instance object directly
      instance.url = `dicomweb:${signedUrl}`;
    } catch (error) {
      console.error(`Error signing URL for ${filePath}:`, error);
    }
  }
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const url = new URL(req.url)
    const patientId = url.searchParams.get('patientId')

    if (!patientId) throw new Error('Missing patientId')

    // --- CACHING LAYER ---
    // If we have a cached response for this URL, return it.
    // Supabase Edge Runtime handles this automatically with Cache-Control headers.

    console.log(`Fetching manifest for: ${patientId}`)

    // Download manifest from R2
    const manifestCommand = new GetObjectCommand({
      Bucket: bucketName,
      Key: `${patientId}/dicom_manifest.json`
    });

    const manifestResponse = await s3Client.send(manifestCommand);
    const manifestText = await manifestResponse.Body?.transformToString() ?? '{}';
    const manifest = JSON.parse(manifestText);

    // --- PARALLEL SIGNING ---
    const signingPromises: Promise<void>[] = [];
    const studies = manifest.studies || []
    for (const study of studies) {
      const seriesList = study.series || []
      for (const series of seriesList) {
        const instances = series.instances || []
        for (const instance of instances) {
          // Add the signing task to our list
          signingPromises.push(signInstanceUrl(patientId, instance));
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