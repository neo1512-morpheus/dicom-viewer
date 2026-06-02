import React, { createContext, useContext } from 'react';

import { useCPROrchestrator } from './useCPROrchestrator';

type CPROrchestratorContextValue = ReturnType<typeof useCPROrchestrator>;

const CPROrchestratorContext = createContext<CPROrchestratorContextValue | null>(null);

interface CPROrchestratorProviderProps {
  children: React.ReactNode;
  servicesManager: AppTypes.ServicesManager;
  commandsManager: AppTypes.CommandsManager;
}

export function CPROrchestratorProvider({
  children,
  servicesManager,
  commandsManager,
}: CPROrchestratorProviderProps): JSX.Element {
  const orchestrator = useCPROrchestrator({
    servicesManager,
    commandsManager,
    sourceViewportId: 'cpr-axial',
    panoWidth: 800,
    panoHeight: 500,
    slabHalfThicknessMm: 15,
    slabSamples: 41,
    aggregation: 'MEAN',
  });

  return (
    <CPROrchestratorContext.Provider value={orchestrator}>
      {children}
    </CPROrchestratorContext.Provider>
  );
}

export function useCPROrchestratorContext(): CPROrchestratorContextValue | null {
  return useContext(CPROrchestratorContext);
}
