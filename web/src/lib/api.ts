export interface WebSocketMessage {
  type: 'status' | 'final_response' | 'error';
  data: any;
}

export const chatAPI = {
  sendMessage: async (data: { message: string; session_id: string }) => {
    // Mock response for now
    return {
      response: `This is a mock response to: ${data.message}`,
      images: [],
      agent_steps: [],
    };
  },
  healthCheck: async () => {
    // Mock health check
    return;
  },
  getImageUrl: (filename: string) => {
    return `/images/${filename}`;
  },
};

export class WebSocketManager {
  constructor(
    sessionId: string,
    onMessage: (message: WebSocketMessage) => void,
    onOpen: () => void,
    onClose: () => void,
    onError: (error: any) => void
  ) {
    // Mock implementation
  }

  connect() {
    // Mock implementation
  }

  disconnect() {
    // Mock implementation
  }

  sendMessage(data: any) {
    // Mock implementation
  }

  isConnected() {
    return false;
  }
}
