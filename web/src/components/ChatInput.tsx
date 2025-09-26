'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Send, Square, Loader2 } from 'lucide-react'
import { Button } from './ui/button'
import { Textarea } from './ui/textarea'
import { cn } from '@/lib/utils'

interface ChatInputProps {
  onSendMessage: (message: string) => void
  isLoading?: boolean
  onStopGeneration?: () => void
  disabled?: boolean
  placeholder?: string
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading = false,
  onStopGeneration,
  disabled = false,
  placeholder = "Ask about ARGO float data..."
}) => {
  const [message, setMessage] = useState('')
  const [isFocused, setIsFocused] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [message])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (message.trim() && !isLoading && !disabled && message.trim().length <= 100) {
      onSendMessage(message.trim())
      setMessage('')

      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleStop = () => {
    if (onStopGeneration) {
      onStopGeneration()
    }
  }

  const canSend = message.trim().length > 0 && !disabled && message.trim().length <= 100
  const showStopButton = isLoading && onStopGeneration

  // Example queries for quick access
  const exampleQueries = [
    "Show temperature data from Bay of Bengal with scatter plots",
    "Create a global map of dissolved oxygen (DOXY) data quality",
    "Find ARGO floats in the North Atlantic with salinity above 35",
    "Generate plots for temperature vs depth in the Indian Ocean",
  ]

  return (
    <div className="relative">
      {/* Example Queries
      {message === '' && !isFocused && (
        <div className="mb-4">
          <p className="text-sm text-muted-foreground mb-2">Try asking:</p>
          <div className="flex flex-wrap gap-2">
            {exampleQueries.map((query, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => setMessage(query)}
                className="text-xs h-auto py-1 px-2 max-w-[200px] truncate"
                disabled={disabled || isLoading}
              >
                {query}
              </Button>
            ))}
          </div>
        </div>
      )} */}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="relative">
        <div className={cn(
          "relative rounded-lg border transition-all duration-200",
          isFocused ? "ring-2 ring-primary ring-offset-2" : "",
          disabled ? "opacity-50" : ""
        )}>
          <Textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder={placeholder}
            disabled={disabled}
            className={cn(
              "min-h-[52px] max-h-[200px] resize-none border-0 shadow-none focus-visible:ring-0 pr-12",
              "custom-scrollbar",
            )}
            style={{ paddingRight: '50px' }}
          />
          
          {/* Action Button */}
          <div className="absolute bottom-2 right-2">
            {showStopButton ? (
              <Button
                type="button"
                size="icon"
                onClick={handleStop}
                className="h-8 w-8 bg-destructive hover:bg-destructive/90"
              >
                <Square className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                type="submit"
                size="icon"
                disabled={!canSend || isLoading}
                className={cn(
                  "h-8 w-8 transition-all duration-200",
                  canSend
                    ? "ocean-gradient shadow-lg shadow-primary/25 hover:shadow-primary/40 scale-100"
                    : "bg-muted-foreground/20 scale-95"
                )}
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            )}
          </div>
        </div>
        
        {/* Character Count & Hints */}
        <div className="flex justify-between items-center mt-2 text-xs text-muted-foreground">
          <div className="flex gap-4">
            <span>Press Enter to send, Shift+Enter for new line</span>
            {isLoading && <span className="text-primary">Processing your request...</span>}
            {message.length > 100 && (
              <span className="text-destructive">Message too long! Max 100 characters.</span>

            )}
          </div>
          <span className={cn(
            "transition-colors",
            message.length > 100 ? "text-destructive" : ""
          )}>
            {message.length}/100
          </span> 
        </div>
      </form>

      {/* Status Messages */}
      {disabled && (
        <div className="absolute inset-0 bg-background/80 backdrop-blur-sm rounded-lg flex items-center justify-center">
          <div className="text-center">
            <p className="text-sm text-muted-foreground">
              Chat is currently disabled
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Please check your connection or try again later
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default ChatInput