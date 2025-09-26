import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  images?: string[]
  agent_steps?: AgentStep[]
  isLoading?: boolean
}

export interface AgentStep {
  action: string
  observation: string
  timestamp: string
}

export interface ChatSession {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: Date
  updatedAt: Date
}

interface ChatStore {
  // Current session
  currentSession: ChatSession | null
  sessions: ChatSession[]
  isLoading: boolean
  error: string | null
  
  // UI state
  sidebarOpen: boolean
  darkMode: boolean
  
  // Actions
  createNewSession: () => string
  selectSession: (sessionId: string) => void
  deleteSession: (sessionId: string) => void
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void
  updateLastMessage: (updates: Partial<ChatMessage>) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  setSidebarOpen: (open: boolean) => void
  toggleDarkMode: () => void
  clearAllSessions: () => void
}

const generateId = () => Math.random().toString(36).substring(2) + Date.now().toString(36)

const createNewChatSession = (): ChatSession => ({
  id: generateId(),
  title: 'New Chat',
  messages: [],
  createdAt: new Date(),
  updatedAt: new Date()
})

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      // Initial state
      currentSession: null,
      sessions: [],
      isLoading: false,
      error: null,
      sidebarOpen: true,
      darkMode: false,

      // Create new session
      createNewSession: () => {
        const newSession = createNewChatSession()
        set((state) => ({
          sessions: [newSession, ...state.sessions],
          currentSession: newSession,
        }))
        return newSession.id
      },

      // Select existing session
      selectSession: (sessionId: string) => {
        const session = get().sessions.find(s => s.id === sessionId)
        if (session) {
          set({ currentSession: session })
        }
      },

      // Delete session
      deleteSession: (sessionId: string) => {
        const { sessions, currentSession } = get()
        const updatedSessions = sessions.filter(s => s.id !== sessionId)
        
        let newCurrentSession = currentSession
        if (currentSession?.id === sessionId) {
          newCurrentSession = updatedSessions[0] || null
        }
        
        set({
          sessions: updatedSessions,
          currentSession: newCurrentSession
        })
      },

      // Add new message
      addMessage: (message) => {
        const { currentSession } = get()
        if (!currentSession) return

        const newMessage: ChatMessage = {
          ...message,
          id: generateId(),
          timestamp: new Date()
        }

        // Update session title based on first user message
        let updatedTitle = currentSession.title
        if (message.role === 'user' && currentSession.messages.length === 0) {
          updatedTitle = message.content.slice(0, 50) + (message.content.length > 50 ? '...' : '')
        }

        const updatedSession = {
          ...currentSession,
          title: updatedTitle,
          messages: [...currentSession.messages, newMessage],
          updatedAt: new Date()
        }

        set((state) => ({
          sessions: state.sessions.map(s => 
            s.id === currentSession.id ? updatedSession : s
          ),
          currentSession: updatedSession
        }))
      },

      // Update the last message (useful for streaming responses)
      updateLastMessage: (updates) => {
        const { currentSession } = get()
        if (!currentSession || currentSession.messages.length === 0) return

        const updatedMessages = [...currentSession.messages]
        const lastIndex = updatedMessages.length - 1
        updatedMessages[lastIndex] = {
          ...updatedMessages[lastIndex],
          ...updates
        }

        const updatedSession = {
          ...currentSession,
          messages: updatedMessages,
          updatedAt: new Date()
        }

        set((state) => ({
          sessions: state.sessions.map(s => 
            s.id === currentSession.id ? updatedSession : s
          ),
          currentSession: updatedSession
        }))
      },

      // Set loading state
      setLoading: (loading) => set({ isLoading: loading }),

      // Set error state
      setError: (error) => set({ error }),

      // UI state
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      toggleDarkMode: () => set((state) => ({ darkMode: !state.darkMode })),

      // Clear all sessions
      clearAllSessions: () => set({
        sessions: [],
        currentSession: null
      })
    }),
    {
      name: 'floatchat-storage',
      // Only persist certain fields
      partialize: (state) => ({
        sessions: state.sessions,
        currentSession: state.currentSession,
        darkMode: state.darkMode,
        sidebarOpen: state.sidebarOpen
      })
    }
  )
)