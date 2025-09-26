// Base API URL - adjust if your FastAPI runs on a different port
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
};

// Export types for use in components
export type { ChatMessage, ChatResponse, AgentStep };