'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Menu, AlertCircle, Wifi, WifiOff, Bot } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import ChatMessage from '@/components/ChatMessage'
import ChatInput from '@/components/ChatInput'
import { AppSidebar } from "@/components/app-sidebar"
import { SidebarProvider } from '@/components/ui/sidebar'
import { useChatStore } from '@/lib/store'
import { chatAPI, WebSocketManager, type WebSocketMessage } from '@/lib/api'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import { SiteHeader } from "@/components/site-header"
import {
  SidebarInset
} from "@/components/ui/sidebar"

export default function ChatPage() {
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const wsManagerRef = useRef<WebSocketManager | null>(null)
  
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [apiConnected, setApiConnected] = useState(true)
  const [wsConnected, setWsConnected] = useState(false)
  
  const {
    currentSession,
    sessions,
    isLoading,
    error,
    darkMode,
    createNewSession,
    addMessage,
    updateLastMessage,
    setLoading,
    setError,
  } = useChatStore()

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [currentSession?.messages, scrollToBottom])

  // Initialize session if none exists
  useEffect(() => {
    if (sessions.length === 0) {
      createNewSession()
    }
  }, [sessions.length, createNewSession])

  // Dark mode handling
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  // WebSocket management
  useEffect(() => {
    if (!currentSession) return

    const handleWsMessage = (message: WebSocketMessage) => {
      switch (message.type) {
        case 'status':
          // Handle status updates (agent thinking, etc.)
          break
        case 'final_response':
          const { response, images = [] } = message.data
          updateLastMessage({
            content: response,
            images,
            isLoading: false
          })
          setLoading(false)
          break
        case 'error':
          setError(message.data.error)
          updateLastMessage({ isLoading: false })
          setLoading(false)
          toast.error('Error processing your request')
          break
      }
    }

    wsManagerRef.current = new WebSocketManager(
      currentSession.id,
      handleWsMessage,
      () => setWsConnected(true),
      () => setWsConnected(false),
      (error) => {
        console.error('WebSocket error:', error)
        setWsConnected(false)
      }
    )

    wsManagerRef.current.connect()

    return () => {
      wsManagerRef.current?.disconnect()
      wsManagerRef.current = null
    }
  }, [currentSession, updateLastMessage, setLoading, setError])

  // Handle sending messages
  const handleSendMessage = async (message: string) => {
    if (!currentSession || isLoading) return

    const sessionId = currentSession.id

    // Add user message
    addMessage({
      role: 'user',
      content: message
    })

    // Add loading assistant message
    addMessage({
      role: 'assistant',
      content: '',
      isLoading: true
    })

    setLoading(true)
    setError(null)

    try {
      // Try WebSocket first, fallback to HTTP
      if (wsManagerRef.current?.isConnected()) {
        wsManagerRef.current.sendMessage({
          type: 'chat',
          message
        })
      } else {
        // Fallback to HTTP API
        const response = await chatAPI.sendMessage({
          message,
          session_id: sessionId
        })

        updateLastMessage({
          content: response.response,
          images: response.images,
          agent_steps: response.agent_steps,
          isLoading: false
        })
      }
    } catch (error) {
      console.error('Failed to send message:', error)
      setError(error instanceof Error ? error.message : 'Failed to send message')
      updateLastMessage({
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        isLoading: false
      })
      toast.error('Failed to send message')
    } finally {
      setLoading(false)
    }
  }

  const handleStopGeneration = () => {
    setLoading(false)
    updateLastMessage({ isLoading: false })
    // In a real implementation, you might want to send a stop signal to the backend
  }

  // Check API health periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await chatAPI.healthCheck()
        setApiConnected(true)
      } catch {
        setApiConnected(false)
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 30000) // Check every 30 seconds

    return () => clearInterval(interval)
  }, [])

  const EmptyState = () => (
  <div className="flex-1 flex items-center justify-center p-8">
    <Card className="max-w-md text-center p-8 bg-card border-border">
      <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
        <Bot className="h-6 w-6 text-primary" />
      </div>
      <h2 className="text-xl font-semibold text-foreground">
        Welcome to FloatChat
      </h2>
      <p className="text-muted-foreground mb-3">
        Your AI assistant for ARGO ocean data exploration and analysis.
        Ask me about ocean temperature, salinity, float locations, and more!
      </p>
      <div className="text-sm text-muted-foreground space-y-1">
        <p>💡 Try asking about specific regions or parameters</p>
        <p>📊 Request visualizations and data plots</p>
        <p>🌊 Explore global ocean data patterns</p>
      </div>
    </Card>
  </div>
  )


  const ConnectionStatus = () => (
    <div className="flex items-center gap-2 text-xs">
      <div className="flex items-center gap-1">
        {apiConnected ? (
          <Wifi className="h-3 w-3 text-green-500" />
        ) : (
          <WifiOff className="h-3 w-3 text-red-500" />
        )}
        <span className={apiConnected ? "text-green-600" : "text-red-600"}>
          API {apiConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
      
      <div className="flex items-center gap-1">
        <div className={cn(
          "w-2 h-2 rounded-full",
          wsConnected ? "bg-green-500" : "bg-gray-400"
        )} />
        <span className={wsConnected ? "text-green-600" : "text-muted-foreground"}>
          {wsConnected ? 'Real-time' : 'HTTP only'}
        </span>
      </div>
    </div>
  )

  return (
    <SidebarProvider
      style={
        {
          "--sidebar-width": "calc(var(--spacing) * 72)",
          "--header-height": "calc(var(--spacing) * 12)",
        } as React.CSSProperties
      }
    >
    <AppSidebar variant="inset" />
    <SidebarInset>
      <SiteHeader />
    

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between p-4 border-b bg-background/95 backdrop-blur">
          <div className="flex items-center gap-3">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden"
            >
              <Menu className="h-4 w-4" />
            </Button>
            
            <div>
              <ConnectionStatus />
            </div>
          </div>

          {currentSession && currentSession.messages.length > 0 && (
            <div className="text-sm text-muted-foreground">
              {currentSession.messages.length} messages
            </div>
          )}
        </header>

        {/* Error Banner */}
        {error && (
          <div className="bg-destructive/10 border-destructive/20 border-b p-3">
            <div className="flex items-center gap-2 text-destructive">
              <AlertCircle className="h-4 w-4" />
              <span className="text-sm">{error}</span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setError(null)}
                className="ml-auto h-6 px-2"
              >
                ×
              </Button>
            </div>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {!currentSession || currentSession.messages.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="pb-4">
              {currentSession.messages.map((message, index) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t bg-background/95 backdrop-blur p-4">
          <div className="max-w-4xl mx-auto">
            <ChatInput
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              onStopGeneration={handleStopGeneration}
              disabled={!apiConnected}
              placeholder={
                !apiConnected 
                  ? "Waiting for API connection..." 
                  : "Ask about ARGO float data..."
              }
            />
          </div>
        </div>
      </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
