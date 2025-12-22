import { id } from './id';
import commandsModule from './commandsModule';
import getPanelModule from './getPanelModule';
import getHangingProtocolModule from './getHangingProtocolModule';
import { cache } from '@cornerstonejs/core';

/**
 * You can remove any of the following modules if you don't need them.
 */
const dynamicVolumeExtension = {
  /**
   * Only required property. Should be a unique value across all extensions.
   * You ID can be anything you want, but it should be unique.
   */
  id,

  /**
   * Perform any pre-registration tasks here. This is called before the extension
   * is registered. Usually we run tasks such as: configuring the libraries
   * (e.g. cornerstone, cornerstoneTools, ...) or registering any services that
   * this extension is providing.
   */
  preRegistration: ({ servicesManager, commandsManager, configuration = {} }) => {
    // 1. Get the RAM approximation (Chrome/Edge only).
    // If undefined (Firefox/Safari), we assume it is NULL/Low-End.
    const deviceMemory = (navigator as any).deviceMemory;

    // 2. Define the Safe Limit (1.5GB) and High Performance Limit (3GB)
    // Note: We use 3GB instead of 4GB to leave room for the OS on 8GB laptops.
    const SAFE_LIMIT = 1.5 * 1024 * 1024 * 1024;
    const HIGH_LIMIT = 3 * 1024 * 1024 * 1024;

    let cacheLimit = SAFE_LIMIT; // DEFAULT TO SAFE MODE

    // 3. Only upgrade to High Limit if we are SURE the device has >= 8GB RAM
    if (deviceMemory && deviceMemory >= 8) {
      cacheLimit = HIGH_LIMIT;
    }

    cache.setMaxCacheSize(cacheLimit);
    console.log(`[Cache] Dynamic Limit Set to: ${(cacheLimit / 1024 / 1024 / 1024).toFixed(1)} GB (Detected RAM: ${deviceMemory || 'Unknown'})`);
  },
  /**
   * PanelModule should provide a list of panels that will be available in OHIF
   * for Modes to consume and render. Each panel is defined by a {name,
   * iconName, iconLabel, label, component} object. Example of a panel module
   * is the StudyBrowserPanel that is provided by the default extension in OHIF.
   */
  getPanelModule,
  /**
   * ViewportModule should provide a list of viewports that will be available in OHIF
   * for Modes to consume and use in the viewports. Each viewport is defined by
   * {name, component} object. Example of a viewport module is the CornerstoneViewport
   * that is provided by the Cornerstone extension in OHIF.
   */
  getHangingProtocolModule,
  /**
   * CommandsModule should provide a list of commands that will be available in OHIF
   * for Modes to consume and use in the viewports. Each command is defined by
   * an object of { actions, definitions, defaultContext } where actions is an
   * object of functions, definitions is an object of available commands, their
   * options, and defaultContext is the default context for the command to run against.
   */
  getCommandsModule: ({ servicesManager, commandsManager, extensionManager }) => {
    return commandsModule({
      servicesManager,
      commandsManager,
      extensionManager,
    });
  },
};

export { dynamicVolumeExtension as default };
