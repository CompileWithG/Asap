// Base API URL - adjust if your FastAPI runs on a different port
const API_BASE_URL = 'http://localhost:8000';

export interface ChatMessage {
  message: string;
  session_id?: string;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  images: string[];
  agent_steps: AgentStep[];
}

export interface AgentStep {
  action: string;
  observation: string;
  timestamp: string;
}

export interface WebSocketMessage {
  type: 'status' | 'final_response' | 'error' | 'agent_step';
  data: any;
}

export const chatAPI = {
  sendMessage: async (data: ChatMessage): Promise<ChatResponse> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(`HTTP ${response.status}: ${errorData.detail || 'Request failed'}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  },

  healthCheck: async (): Promise<{ status: string; timestamp: string }> => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  getImageUrl: (filename: string): string => {
    return `${API_BASE_URL}/api/images/${filename}`;
  },

  // Utility function to test connection
  testConnection: async (): Promise<boolean> => {
    try {
      await chatAPI.healthCheck();
      return true;
    } catch {
      return false;
    }
  },
};

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private sessionId: string;
  private onMessage: (message: WebSocketMessage) => void;
  private onOpen: () => void;
  private onClose: () => void;
  private onError: (error: any) => void;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 1000;

  constructor(
    sessionId: string,
    onMessage: (message: WebSocketMessage) => void,
    onOpen: () => void,
    onClose: () => void,
    onError: (error: any) => void
  ) {
    this.sessionId = sessionId;
    this.onMessage = onMessage;
    this.onOpen = onOpen;
    this.onClose = onClose;
    this.onError = onError;
  }

  connect(): void {
    try {
      // Convert HTTP URL to WebSocket URL
      const wsUrl = `ws://localhost:8000/ws/${this.sessionId}`;
      
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = (event) => {
        console.log('WebSocket connected:', event);
        this.reconnectAttempts = 0;
        this.onOpen();
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.onMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
          this.onError(error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket closed:', event);
        this.ws = null;
        this.onClose();
        
        // Attempt to reconnect if it wasn't a manual close
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.attemptReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.onError(error);
      };

    } catch (error) {
      console.error('Error creating WebSocket:', error);
      this.onError(error);
    }
  }

  private attemptReconnect(): void {
    this.reconnectAttempts++;
    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
    
    setTimeout(() => {
      if (this.reconnectAttempts <= this.maxReconnectAttempts) {
        this.connect();
      }
    }, this.reconnectDelay * this.reconnectAttempts); // Exponential backoff
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect');
      this.ws = null;
    }
  }

  sendMessage(data: { type: string; message: string }): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(data));
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
        this.onError(error);
      }
    } else {
      console.warn('WebSocket is not connected. Message not sent:', data);
      this.onError(new Error('WebSocket is not connected'));
    }
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  getReadyState(): number | null {
    return this.ws ? this.ws.readyState : null;
  }
}

// Utility functions for development/testing
export const devUtils = {
  // Test the API connection
  testAPI: async () => {
    try {
      console.log('Testing API connection...');
      const health = await chatAPI.healthCheck();
      console.log('✅ API Health Check passed:', health);
      
      // Test a simple chat message
      const testResponse = await chatAPI.sendMessage({
        message: 'Hello, this is a test message',
        session_id: 'test-session'
      });
      console.log('✅ Chat API test passed:', testResponse);
      
      return true;
    } catch (error) {
      console.error('❌ API test failed:', error);
      return false;
    }
  },

  // Test WebSocket connection
  testWebSocket: (sessionId: string = 'test-ws-session') => {
    return new Promise((resolve, reject) => {
      const testWS = new WebSocketManager(
        sessionId,
        (message) => {
          console.log('📨 WebSocket message received:', message);
        },
        () => {
          console.log('✅ WebSocket connected successfully');
          
          // Send a test message
          testWS.sendMessage({
            type: 'chat',
            message: 'Hello via WebSocket!'
          });
          
          // Clean up after a delay
          setTimeout(() => {
            testWS.disconnect();
            resolve(true);
          }, 2000);
        },
        () => {
          console.log('WebSocket disconnected');
        },
        (error) => {
          console.error('❌ WebSocket error:', error);
          reject(error);
        }
      );

      testWS.connect();

      // Timeout after 10 seconds
      setTimeout(() => {
        testWS.disconnect();
        reject(new Error('WebSocket test timeout'));
      }, 10000);
    });
  }
};

// Export types for use in components
export type { ChatMessage, ChatResponse, AgentStep, WebSocketMessage };