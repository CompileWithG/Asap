'use client'

import React, { useState } from 'react'
import Image from 'next/image'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { 
  Copy, 
  Check, 
  User, 
  Bot, 
  Image as ImageIcon, 
  Download,
  ExternalLink,
  AlertCircle 
} from 'lucide-react'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { cn, copyToClipboard, formatTime } from '@/lib/utils'
import { chatAPI } from '@/lib/api'
import type { ChatMessage as ChatMessageType } from '@/lib/store'

interface ChatMessageProps {
  message: ChatMessageType
}

interface ImageDisplayProps {
  filename: string
  alt?: string
}

const ImageDisplay: React.FC<ImageDisplayProps> = ({ filename, alt }) => {
  const [imageLoaded, setImageLoaded] = useState(false)
  const [imageError, setImageError] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)

  const imageUrl = chatAPI.getImageUrl(filename)

  const handleDownload = async () => {
    try {
      const response = await fetch(imageUrl)
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to download image:', error)
    }
  }

  if (imageError) {
    return (
      <Card className="p-4 border-destructive/20 bg-destructive/5">
        <div className="flex items-center gap-2 text-destructive">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">Failed to load image: {filename}</span>
        </div>
      </Card>
    )
  }

  return (
    <>
      <Card className="overflow-hidden bg-muted/20">
        <div className="relative group">
          {!imageLoaded && (
            <div className="h-64 bg-muted animate-pulse flex items-center justify-center">
              <ImageIcon className="h-8 w-8 text-muted-foreground" />
            </div>
          )}
          
          <Image
            src={imageUrl}
            alt={alt || filename}
            width={800}
            height={600}
            className={cn(
              "w-full h-auto transition-opacity duration-300 cursor-zoom-in",
              imageLoaded ? "opacity-100" : "opacity-0 absolute inset-0"
            )}
            onLoad={() => setImageLoaded(true)}
            onError={() => setImageError(true)}
            onClick={() => setIsFullscreen(true)}
            priority={false}
          />
          
          {imageLoaded && (
            <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="flex gap-1">
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={handleDownload}
                  className="h-8 w-8 p-0"
                >
                  <Download className="h-3 w-3" />
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => window.open(imageUrl, '_blank')}
                  className="h-8 w-8 p-0"
                >
                  <ExternalLink className="h-3 w-3" />
                </Button>
              </div>
            </div>
          )}
        </div>
        
        <div className="p-3 border-t bg-background/50">
          <p className="text-xs text-muted-foreground truncate" title={filename}>
            {filename}
          </p>
        </div>
      </Card>

      {/* Fullscreen Modal */}
      {isFullscreen && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setIsFullscreen(false)}
        >
          <div className="relative max-w-7xl max-h-full">
            <Image
              src={imageUrl}
              alt={alt || filename}
              width={1200}
              height={900}
              className="max-w-full max-h-full object-contain"
              onClick={(e) => e.stopPropagation()}
            />
            <Button
              size="sm"
              variant="secondary"
              className="absolute top-4 right-4"
              onClick={() => setIsFullscreen(false)}
            >
              Close
            </Button>
          </div>
        </div>
      )}
    </>
  )
}

const TypingIndicator: React.FC = () => (
  <div className="flex items-center gap-2 text-muted-foreground">
    <div className="typing-indicator">
      <div className="typing-dot"></div>
      <div className="typing-dot"></div>
      <div className="typing-dot"></div>
    </div>
    <span className="text-sm">Agent is thinking...</span>
  </div>
)

const CodeBlock: React.FC<{ language: string; children: string }> = ({ language, children }) => (
  <SyntaxHighlighter
    style={tomorrow}
    language={language}
    PreTag="div"
    className="rounded-md"
  >
    {children}
  </SyntaxHighlighter>
)

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'
  const isAssistant = message.role === 'assistant'

  const handleCopy = async () => {
    const success = await copyToClipboard(message.content)
    if (success) {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <div className={cn(
      "flex gap-3 p-4 message-enter",
      isUser ? "flex-row-reverse" : "flex-row"
    )}>
      {/* Avatar */}
      <div className={cn(
        "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium",
        isUser 
          ? "bg-primary text-primary-foreground" 
          : "ocean-gradient text-white"
      )}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      {/* Message Content */}
      <div className={cn(
        "flex-1 min-w-0",
        isUser ? "text-right" : "text-left"
      )}>
        {/* Message Header */}
        <div className="flex items-center gap-2 mb-1">
          {!isUser && <span className="text-sm font-medium">FloatChat Assistant</span>}
          <span className="text-xs text-muted-foreground">
            {formatTime(message.timestamp)}
          </span>
          {message.content && (
            <Button
              size="sm"
              variant="ghost"
              onClick={handleCopy}
              className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
            >
              {copied ? (
                <Check className="h-3 w-3 text-green-600" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </Button>
          )}
        </div>

        {/* Loading State */}
        {message.isLoading && isAssistant && (
          <div className="mb-3">
            <TypingIndicator />
          </div>
        )}

        {/* Message Text */}
        {message.content && (
          <Card className={cn(
            "p-4 group",
            isUser 
              ? "bg-primary text-primary-foreground ml-auto max-w-[80%]" 
              : "bg-muted/50 mr-auto max-w-[90%]"
          )}>
            {isUser ? (
              <p className="text-sm">{message.content}</p>
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                className="prose prose-sm max-w-none dark:prose-invert"
                components={{
                  code({ node, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || ''); const inline = !match;
                    return !inline && match ? (
                      <CodeBlock language={match[1]}>
                        {String(children).replace(/\n$/, '')}
                      </CodeBlock>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    )
                  }
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}
          </Card>
        )}

        {/* Images */}
        {message.images && message.images.length > 0 && (
          <div className="mt-3 space-y-3">
            {message.images.map((image, index) => (
              <ImageDisplay
                key={index}
                filename={image}
                alt={`Generated visualization ${index + 1}`}
              />
            ))}
          </div>
        )}

        {/* Agent Steps (Debug Info) */}
        {message.agent_steps && message.agent_steps.length > 0 && (
          <details className="mt-3">
            <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
              Show agent execution steps ({message.agent_steps.length})
            </summary>
            <div className="mt-2 space-y-2 text-xs">
              {message.agent_steps.map((step, index) => (
                <Card key={index} className="p-3 bg-muted/30">
                  <div className="font-medium text-primary mb-1">Action:</div>
                  <pre className="whitespace-pre-wrap mb-2 text-muted-foreground">
                    {step.action}
                  </pre>
                  <div className="font-medium text-primary mb-1">Observation:</div>
                  <pre className="whitespace-pre-wrap text-muted-foreground">
                    {step.observation}
                  </pre>
                </Card>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}

export default ChatMessage