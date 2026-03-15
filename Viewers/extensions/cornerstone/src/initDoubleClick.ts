import { eventTarget, EVENTS, getEnabledElement } from '@cornerstonejs/core';
import { Enums } from '@cornerstonejs/tools';
import { CommandsManager, CustomizationService, Types } from '@ohif/core';
import { findNearbyToolData } from './utils/findNearbyToolData';

const cs3DToolsEvents = Enums.Events;

const DEFAULT_DOUBLE_CLICK = {
  doubleClick: {
    commandName: 'toggleOneUp',
    commandOptions: {},
  },
};

/**
 * Generates a double click event name, consisting of:
 *    * alt when the alt key is down
 *    * ctrl when the cctrl key is down
 *    * shift when the shift key is down
 *    * 'doubleClick'
 */
function getDoubleClickEventName(evt: CustomEvent) {
  const nameArr = [];
  if (evt.detail.event.altKey) {
    nameArr.push('alt');
  }
  if (evt.detail.event.ctrlKey) {
    nameArr.push('ctrl');
  }
  if (evt.detail.event.shiftKey) {
    nameArr.push('shift');
  }
  nameArr.push('doubleClick');
  return nameArr.join('');
}

export type initDoubleClickArgs = {
  customizationService: CustomizationService;
  commandsManager: CommandsManager;
  viewportGridService: {
    getState: () => {
      viewports:
        | Map<
            string,
            {
              viewportOptions?: {
                viewportId?: string;
              };
            }
          >
        | Record<
            string,
            {
              viewportOptions?: {
                viewportId?: string;
              };
            }
          >;
    };
  };
};

const CPR_LOGICAL_VIEWPORT_IDS = new Set(['cpr-axial', 'cpr-pano', 'cpr-crosssection']);

function initDoubleClick({
  customizationService,
  commandsManager,
  viewportGridService,
}: initDoubleClickArgs): void {
  const cornerstoneViewportHandleDoubleClick = (evt: CustomEvent) => {
    // Do not allow double click on a tool.
    const nearbyToolData = findNearbyToolData(commandsManager, evt);
    if (nearbyToolData) {
      return;
    }

    const eventName = getDoubleClickEventName(evt);

    // Allows for the customization of the double click on a viewport.
    const customizations =
      customizationService.get('cornerstoneViewportClickCommands') || DEFAULT_DOUBLE_CLICK;

    const toRun = customizations[eventName];

    if (!toRun) {
      return;
    }

    const clickedElement = evt.detail?.element as HTMLDivElement | undefined;
    const clickedViewportId = clickedElement
      ? getEnabledElement(clickedElement)?.viewport?.id
      : undefined;
    const viewports = viewportGridService.getState().viewports;
    const logicalViewportId =
      clickedViewportId &&
      (typeof (viewports as Map<string, { viewportOptions?: { viewportId?: string } }>).get ===
      'function'
        ? (viewports as Map<string, { viewportOptions?: { viewportId?: string } }>).get(
            clickedViewportId
          )?.viewportOptions?.viewportId
        : (
            viewports as Record<string, { viewportOptions?: { viewportId?: string } }>
          )?.[clickedViewportId]?.viewportOptions?.viewportId);
    const commandToRun =
      toRun.commandName === 'toggleOneUp' &&
      logicalViewportId &&
      CPR_LOGICAL_VIEWPORT_IDS.has(logicalViewportId)
        ? {
            ...toRun,
            commandName: 'toggleViewportOneUpAware',
          }
        : toRun;

    if (clickedViewportId) {
      commandsManager.runCommand('setViewportActive', { viewportId: clickedViewportId });
    }

    commandsManager.run(commandToRun);
  };

  function elementEnabledHandler(evt: CustomEvent) {
    const { element } = evt.detail;

    element.addEventListener(
      cs3DToolsEvents.MOUSE_DOUBLE_CLICK,
      cornerstoneViewportHandleDoubleClick
    );
  }

  function elementDisabledHandler(evt: CustomEvent) {
    const { element } = evt.detail;

    element.removeEventListener(
      cs3DToolsEvents.MOUSE_DOUBLE_CLICK,
      cornerstoneViewportHandleDoubleClick
    );
  }

  eventTarget.addEventListener(EVENTS.ELEMENT_ENABLED, elementEnabledHandler.bind(null));

  eventTarget.addEventListener(EVENTS.ELEMENT_DISABLED, elementDisabledHandler.bind(null));
}

export default initDoubleClick;
