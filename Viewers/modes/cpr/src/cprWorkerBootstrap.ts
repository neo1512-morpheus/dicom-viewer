interface BootstrapLikeMessage {
  type?: unknown;
  requestId?: unknown;
}

interface BootstrapReadyMessage {
  type: 'BOOTSTRAP_READY';
  requestId: string;
}

interface DisposeSuccessMessage {
  type: 'DISPOSE_SUCCESS';
  requestId: string;
}

interface ErrorMessage {
  type: 'ERROR';
  requestId: string;
  message: string;
  stage?: string;
}

interface WorkerLifecycleMessage {
  type: 'WORKER_LIFECYCLE';
  scope: 'bootstrap';
  stage: string;
  detail?: Record<string, unknown>;
}

let workerImplLoaded = false;
let workerImplPromise: Promise<void> | null = null;
let bootstrapLifecycleSequence = 0;

function resolveBootstrapRequestId(input: unknown): string {
  return typeof (input as BootstrapLikeMessage | null | undefined)?.requestId === 'string'
    ? String((input as BootstrapLikeMessage).requestId)
    : 'unknown-request';
}

function postBootstrapMessage(
  message: BootstrapReadyMessage | DisposeSuccessMessage | ErrorMessage | WorkerLifecycleMessage
): void {
  // eslint-disable-next-line no-restricted-globals
  (self as unknown as Worker).postMessage(message);
}

function logBootstrapLifecycle(stage: string, detail?: Record<string, unknown>): void {
  console.log(
    '[CPR-WORKER-BOOTSTRAP-LIFECYCLE-JSON]',
    JSON.stringify({
      stage,
      ...(detail ?? {}),
    })
  );
}

function emitBootstrapLifecycle(stage: string, detail?: Record<string, unknown>): void {
  bootstrapLifecycleSequence += 1;
  const lifecycleDetail = {
    sequence: bootstrapLifecycleSequence,
    ...(detail ?? {}),
  };
  logBootstrapLifecycle(stage, lifecycleDetail);
  console.log(
    '[CPR-WORKER-BOOTSTRAP-PREPOST-JSON]',
    JSON.stringify({
      stage,
      ...lifecycleDetail,
    })
  );
  postBootstrapMessage({
    type: 'WORKER_LIFECYCLE',
    scope: 'bootstrap',
    stage,
    detail: lifecycleDetail,
  });
}

function postBootstrapError(requestId: string, message: string, stage: string): void {
  postBootstrapMessage({
    type: 'ERROR',
    requestId,
    message,
    stage,
  });
}

async function ensureWorkerImplementationLoaded(): Promise<void> {
  if (workerImplLoaded) {
    return;
  }

  if (!workerImplPromise) {
    emitBootstrapLifecycle('worker-implementation-import-started');
    workerImplPromise = import('./cprWorker')
      .then(() => {
        workerImplLoaded = true;
        emitBootstrapLifecycle('worker-implementation-import-succeeded');
      })
      .catch(error => {
        workerImplPromise = null;
        console.error('[CPR-WORKER-BOOTSTRAP-ERROR-JSON]', {
          stage: 'worker-implementation-import-failed',
          message: error instanceof Error ? error.message : String(error),
        });
        throw error;
      });
  }

  await workerImplPromise;
}

const bootstrapOnMessage = async (event: MessageEvent<BootstrapLikeMessage>) => {
  const input = event.data;
  const requestId = resolveBootstrapRequestId(input);
  const requestType = typeof input?.type === 'string' ? input.type : 'unknown';

  try {
    emitBootstrapLifecycle('message-received', {
      requestType,
      requestId,
      workerImplLoaded,
    });

    if (input?.type === 'BOOTSTRAP_CHECK') {
      emitBootstrapLifecycle('bootstrap-check-received', {
        requestId,
      });
      postBootstrapMessage({
        type: 'BOOTSTRAP_READY',
        requestId,
      });
      emitBootstrapLifecycle('bootstrap-ready-sent', {
        requestId,
      });
      void ensureWorkerImplementationLoaded().catch(error => {
        console.error('[CPR-WORKER-BOOTSTRAP-ERROR-JSON]', {
          stage: 'bootstrap-check-preload-failed',
          requestId,
          message: error instanceof Error ? error.message : String(error),
        });
      });
      return;
    }

    if (input?.type === 'DISPOSE' && !workerImplLoaded) {
      emitBootstrapLifecycle('dispose-received-before-impl-load', {
        requestId,
      });
      postBootstrapMessage({
        type: 'DISPOSE_SUCCESS',
        requestId,
      });
      emitBootstrapLifecycle('dispose-success-sent-before-impl-load', {
        requestId,
      });
      return;
    }

    await ensureWorkerImplementationLoaded();
    const delegatedOnMessage = self.onmessage;
    if (typeof delegatedOnMessage !== 'function' || delegatedOnMessage === bootstrapOnMessage) {
      throw new Error(
        '[cprWorkerBootstrap] Worker implementation did not install an onmessage handler.'
      );
    }

    emitBootstrapLifecycle('delegating-to-worker-implementation', {
      requestType,
      requestId,
    });
    await delegatedOnMessage.call(self, event as unknown as MessageEvent);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('[CPR-WORKER-BOOTSTRAP-ERROR-JSON]', {
      stage: requestType,
      requestId,
      message,
    });
    if (requestId !== 'unknown-request') {
      postBootstrapError(requestId, `[cprWorkerBootstrap] ${message}`, requestType);
    }
  }
};

// eslint-disable-next-line no-restricted-globals
self.addEventListener('error', event => {
  emitBootstrapLifecycle('worker-global-error-event', {
    message: event.message ?? null,
    filename: event.filename ?? null,
    lineno: Number.isFinite(event.lineno) ? event.lineno : null,
    colno: Number.isFinite(event.colno) ? event.colno : null,
    errorMessage: event.error instanceof Error ? event.error.message : null,
  });
});

// eslint-disable-next-line no-restricted-globals
self.addEventListener('messageerror', event => {
  emitBootstrapLifecycle('worker-messageerror-event', {
    origin: event.origin ?? null,
    lastEventId: event.lastEventId ?? null,
    dataType: event.data == null ? null : typeof event.data,
  });
});

// eslint-disable-next-line no-restricted-globals
self.addEventListener('unhandledrejection', event => {
  const reason =
    event.reason instanceof Error
      ? {
          name: event.reason.name,
          message: event.reason.message,
        }
      : {
          message: String(event.reason),
        };
  emitBootstrapLifecycle('worker-unhandled-rejection-event', reason);
});

emitBootstrapLifecycle('worker-bootstrap-script-loaded', {
  locationHref:
    typeof self.location === 'object' && typeof self.location.href === 'string'
      ? self.location.href
      : null,
});

// eslint-disable-next-line no-restricted-globals
self.onmessage = bootstrapOnMessage;
