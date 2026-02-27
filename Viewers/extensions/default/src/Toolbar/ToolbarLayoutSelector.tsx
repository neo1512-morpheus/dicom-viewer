import React, { useEffect, useState, useCallback, useRef } from 'react';
import PropTypes from 'prop-types';
import { LayoutSelector as OHIFLayoutSelector, ToolbarButton, LayoutPreset } from '@ohif/ui';

const defaultCommonPresets = [
  {
    icon: 'layout-common-1x1',
    commandOptions: {
      numRows: 1,
      numCols: 1,
    },
  },
  {
    icon: 'layout-common-1x2',
    commandOptions: {
      numRows: 1,
      numCols: 2,
    },
  },
  {
    icon: 'layout-common-2x2',
    commandOptions: {
      numRows: 2,
      numCols: 2,
    },
  },
  {
    icon: 'layout-common-2x3',
    commandOptions: {
      numRows: 2,
      numCols: 3,
    },
  },
];

const _areSelectorsValid = (hp, displaySets, hangingProtocolService) => {
  if (!hp.displaySetSelectors || Object.values(hp.displaySetSelectors).length === 0) {
    return true;
  }

  return hangingProtocolService.areRequiredSelectorsValid(
    Object.values(hp.displaySetSelectors),
    displaySets[0]
  );
};

// Hardcoded list of 6 advanced presets for the layout menu
// NOTE: reset: true bypasses viewportGridStore cache to prevent race condition bugs
const generateAdvancedPresets = () => {
  return [
    {
      icon: 'layout-common-1x1',
      title: 'Axial',
      evaluate: 'evaluate.mpr',
      commandOptions: { protocolId: 'hpAxial', reset: true },
      disabled: false,
    },
    {
      icon: 'layout-common-1x1',
      title: 'Coronal',
      evaluate: 'evaluate.mpr',
      commandOptions: { protocolId: 'hpCoronal', reset: true },
      disabled: false,
    },
    {
      icon: 'layout-common-1x1',
      title: 'Sagittal',
      evaluate: 'evaluate.mpr',
      commandOptions: { protocolId: 'hpSagittal', reset: true },
      disabled: false,
    },
    {
      icon: 'layout-advanced-3d-only',
      title: '3D Only',
      evaluate: 'evaluate.mpr',
      commandOptions: { protocolId: 'hp3D', reset: true },
      disabled: false,
    },
    {
      icon: 'layout-advanced-3d-four-up',
      title: '3D Four Up',
      evaluate: 'evaluate.mpr',
      commandOptions: { protocolId: 'fourUp', reset: true },
      disabled: false,
    },
    {
      icon: 'layout-advanced-mpr',
      title: 'MPR',
      evaluate: 'evaluate.mpr',
      commandOptions: { protocolId: 'mpr', reset: true },
      disabled: false,
    },
    {
      icon: 'layout-advanced-mpr',
      title: 'CPR',
      evaluate: 'evaluate.mpr',
      commandOptions: { protocolId: 'cpr', stageIndex: 0, reset: true },
      disabled: false,
    },
  ];
};

function ToolbarLayoutSelectorWithServices({
  commandsManager,
  servicesManager,
  ...props
}: withAppTypes) {
  const [isDisabled, setIsDisabled] = useState(false);

  const handleMouseEnter = () => {
    setIsDisabled(false);
  };

  const onSelection = useCallback(props => {
    commandsManager.run({
      commandName: 'setViewportGridLayout',
      commandOptions: { ...props },
    });
    setIsDisabled(true);
  }, []);

  // =====================================================
  // LARGE SCAN WARNING for 3D layouts
  // =====================================================
  const LARGE_SCAN_THRESHOLD = 300;
  // These are the actual protocol IDs from the hanging protocols folder
  const VOLUME_PROTOCOLS = [
    'mpr', '3d', 'volume', 'axial', 'sagittal', 'coronal', // general terms
    'fourup', 'fourUp', // 3D four up
    'cpr', // CPR setup/view stages
    'only3d', // 3D only
    'main3d', // 3D main
    'primary3d', // 3D primary
    'primaryaxial', // Axial primary
    'mprand3dvolumeviewport', // MPR and 3D volume
  ];

  const onSelectionPreset = useCallback(presetProps => {
    const { displaySetService, viewportGridService, uiModalService } =
      servicesManager.services;

    const protocolId = presetProps.protocolId?.toLowerCase() || '';
    const isVolumeLayout = VOLUME_PROTOCOLS.some(vp => protocolId.includes(vp));

    // Get current displaySets to check slice count
    const viewportId = viewportGridService.getActiveViewportId();
    let totalSlices = 0;

    if (viewportId) {
      const displaySetUIDs = viewportGridService.getDisplaySetsUIDsForViewport(viewportId);
      if (displaySetUIDs) {
        displaySetUIDs.forEach(uid => {
          const ds = displaySetService.getDisplaySetByUID(uid);
          if (ds) {
            totalSlices += ds.numImageFrames || ds.instances?.length || 0;
          }
        });
      }
    }

    console.log(`[Layout] Selected: ${protocolId}, isVolume: ${isVolumeLayout}, slices: ${totalSlices}`);

    // If 3D layout AND large scan, show warning
    if (isVolumeLayout && totalSlices > LARGE_SCAN_THRESHOLD) {
      console.log('[Layout] LARGE SCAN - Showing warning modal');

      uiModalService.show({
        content: LargeScanWarningModal,
        contentProps: {
          totalSlices,
          onProceed: () => {
            console.log('[Layout] User clicked PROCEED');
            uiModalService.hide();
            commandsManager.run({
              commandName: 'setHangingProtocol',
              commandOptions: { ...presetProps },
            });
            setIsDisabled(true);
          },
          onUse2D: () => {
            console.log('[Layout] User clicked STAY IN 2D');
            uiModalService.hide();
            // DO NOTHING - stay in current layout
            setIsDisabled(true);
          },
        },
        title: '',
        shouldCloseOnEsc: false,
        shouldCloseOnOverlayClick: false,
        closeButton: false,
        containerDimensions: 'max-w-lg',
        // Style outer wrapper with border/shadow, content will override inner background
        customClassName: 'border-2 border-slate-500 shadow-2xl rounded-xl overflow-hidden',
      });
      return;
    }

    // Normal case - no warning needed
    console.log('[Layout] Small scan or 2D layout - proceeding normally');
    commandsManager.run({
      commandName: 'setHangingProtocol',
      commandOptions: { ...presetProps },
    });
    setIsDisabled(true);
  }, [servicesManager, commandsManager]);

  return (
    <div onMouseEnter={handleMouseEnter}>
      <LayoutSelector
        {...props}
        onSelection={onSelection}
        onSelectionPreset={onSelectionPreset}
        servicesManager={servicesManager}
        tooltipDisabled={isDisabled}
      />
    </div>
  );
}

// Warning modal component for large scans
function LargeScanWarningModal({ totalSlices, onProceed, onUse2D }) {
  return (
    <div style={{
      backgroundColor: '#1e293b',
      padding: '24px',
      margin: '-8px -20px -20px -20px',
      display: 'flex',
      flexDirection: 'column' as const,
      textAlign: 'center' as const,
    }}>
      <h3 style={{ color: '#fbbf24', marginBottom: '15px', fontSize: '18px' }}>
        ⚠️ Large 3D Scan Detected
      </h3>
      <p style={{ marginBottom: '10px', color: '#f1f5f9' }}>
        This scan has <strong style={{ color: '#ff5722' }}>{totalSlices} slices</strong> which requires significant GPU memory.
      </p>
      <p style={{ marginBottom: '25px', color: '#f1f5f9', fontSize: '14px' }}>
        If you don't have a powerful graphics card (1GB+ VRAM), this may cause the viewer to crash or freeze.
      </p>
      <div style={{ display: 'flex', gap: '15px', justifyContent: 'center' }}>
        <button
          onClick={onProceed}
          style={{
            padding: '12px 24px',
            backgroundColor: '#ff5722',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '14px',
          }}
        >
          Proceed Anyway
          <br />
          <span style={{ fontSize: '12px', fontWeight: 'normal' }}>(Risky)</span>
        </button>
        <button
          onClick={onUse2D}
          style={{
            padding: '12px 24px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '14px',
          }}
        >
          Stay in 2D Mode
          <br />
          <span style={{ fontSize: '12px', fontWeight: 'normal' }}>(Safe)</span>
        </button>
      </div>
    </div >
  );
}


function LayoutSelector({
  rows,
  columns,
  className,
  onSelection,
  onSelectionPreset,
  servicesManager,
  tooltipDisabled,
  ...rest
}: withAppTypes) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  const { customizationService, displaySetService, viewportGridService } = servicesManager.services;
  const commonPresets = customizationService.get('commonPresets') || defaultCommonPresets;
  const advancedPresets =
    customizationService.get('advancedPresets') || generateAdvancedPresets({ servicesManager });

  const closeOnOutsideClick = event => {
    if (isOpen && dropdownRef.current) {
      setIsOpen(false);
    }
  };

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    setTimeout(() => {
      window.addEventListener('click', closeOnOutsideClick);
    }, 0);
    return () => {
      window.removeEventListener('click', closeOnOutsideClick);
      dropdownRef.current = null;
    };
  }, [isOpen]);

  const onInteractionHandler = () => {
    setIsOpen(!isOpen);
  };
  const DropdownContent = isOpen ? OHIFLayoutSelector : null;

  return (
    <ToolbarButton
      id="Layout"
      label="Layout"
      icon="tool-layout"
      onInteraction={onInteractionHandler}
      className={className}
      rounded={rest.rounded}
      disableToolTip={tooltipDisabled}
      dropdownContent={
        DropdownContent !== null && (
          <div
            className="flex"
            ref={dropdownRef}
          >
            <div className="bg-secondary-dark flex flex-col gap-2.5 p-2">
              <div className="text-aqua-pale text-xs">Common</div>

              <div className="flex gap-4">
                {commonPresets.map((preset, index) => (
                  <LayoutPreset
                    key={index}
                    classNames="hover:bg-primary-dark group p-1 cursor-pointer"
                    icon={preset.icon}
                    commandOptions={preset.commandOptions}
                    onSelection={onSelection}
                  />
                ))}
              </div>

              <div className="h-[2px] bg-black"></div>

              <div className="text-aqua-pale text-xs">Advanced</div>

              <div className="flex flex-col gap-2.5">
                {advancedPresets.map((preset, index) => {
                  // Check if the current viewport allows this preset (e.g. is it a valid reconstructable volume?)
                  // This mirrors the logic from evaluate.mpr in getToolbarModule.tsx
                  let isPresetDisabled = preset.disabled || false;

                  if (preset.evaluate === 'evaluate.mpr') {
                    const viewportId = viewportGridService.getActiveViewportId();
                    const displaySetUIDs = viewportGridService.getDisplaySetsUIDsForViewport(viewportId);

                    if (displaySetUIDs?.length) {
                      const displaySets = displaySetUIDs.map(uid => displaySetService.getDisplaySetByUID(uid));
                      const areReconstructable = displaySets.every(ds => ds?.isReconstructable);
                      if (!areReconstructable) {
                        isPresetDisabled = true;
                      }
                    }
                  }

                  return (
                    <LayoutPreset
                      key={index + commonPresets.length}
                      classNames={`hover:bg-primary-dark group flex gap-2 p-1 ${isPresetDisabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                      icon={preset.icon}
                      title={preset.title}
                      disabled={isPresetDisabled}
                      commandOptions={preset.commandOptions}
                      onSelection={onSelectionPreset}
                    />
                  );
                })}
              </div>
            </div>

            <div className="bg-primary-dark flex flex-col gap-2.5 border-l-2 border-solid border-black  p-2">
              <div className="text-aqua-pale text-xs">Custom</div>
              <DropdownContent
                rows={rows}
                columns={columns}
                onSelection={onSelection}
              />
              <p className="text-aqua-pale text-xs leading-tight">
                Hover to select <br></br>rows and columns <br></br> Click to apply
              </p>
            </div>
          </div>
        )
      }
      isActive={isOpen}
      type="toggle"
    />
  );
}

LayoutSelector.propTypes = {
  rows: PropTypes.number,
  columns: PropTypes.number,
  onLayoutChange: PropTypes.func,
  servicesManager: PropTypes.object.isRequired,
};

LayoutSelector.defaultProps = {
  columns: 4,
  rows: 3,
  onLayoutChange: () => { },
};

export default ToolbarLayoutSelectorWithServices;
