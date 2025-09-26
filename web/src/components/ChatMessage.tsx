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
  AlertCircle,
  Map,
  Maximize2,
  X
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

interface MapDisplayProps {
  filename: string
  alt?: string
}

const MapDisplay: React.FC<MapDisplayProps> = ({ filename, alt }) => {
  const [mapLoaded, setMapLoaded] = useState(false)
  const [mapError, setMapError] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)

  const mapUrl = chatAPI.getImageUrl(filename).replace('/api', '');

  
  // Add debugging
  console.log('MapDisplay - filename:', filename)
  console.log('MapDisplay - mapUrl:', mapUrl)

  const handleDownload = async () => {
    try {
      const response = await fetch(mapUrl)
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
      console.error('Failed to download map:', error)
    }
  }

  const handleIframeError = (e: React.SyntheticEvent<HTMLIFrameElement>) => {
    console.error('Iframe error:', e)
    setMapError(true)
  }

  const handleIframeLoad = (e: React.SyntheticEvent<HTMLIFrameElement>) => {
    console.log('Iframe loaded successfully:', mapUrl)
    setMapLoaded(true)
    
    // Additional check for iframe content
    const iframe = e.currentTarget
    try {
      // This might fail due to CORS, but worth trying
      if (iframe.contentDocument) {
        console.log('Iframe content accessible')
      }
    } catch (err) {
      console.log('Iframe content not accessible (expected for cross-origin)')
    }
  }

  if (mapError) {
    return (
      <Card className="p-4 border-destructive/20 bg-destructive/5">
        <div className="flex items-center gap-2 text-destructive">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">Failed to load map: {filename}</span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => window.open(mapUrl, '_blank')}
            className="ml-auto"
          >
            Open Directly
          </Button>
        </div>
      </Card>
    )
  }

  return (
    <>
      <Card className="overflow-hidden bg-blue-50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800">
        <div className="relative group">
          <div className="flex items-center justify-between p-3 bg-blue-100 dark:bg-blue-900/30 border-b border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-2">
              <Map className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                Interactive Map
              </span>
            </div>
            <div className="text-xs text-blue-600 dark:text-blue-400">
              {mapLoaded ? 'Loaded' : 'Loading...'}
            </div>
          </div>

          {!mapLoaded && (
            <div className="h-96 bg-muted animate-pulse flex flex-col items-center justify-center gap-2">
              <Map className="h-8 w-8 text-muted-foreground animate-pulse" />
              <span className="text-sm text-muted-foreground">Loading map...</span>
              <span className="text-xs text-muted-foreground">{filename}</span>
            </div>
          )}
          
          <iframe
            key={mapUrl} // Force re-render if URL changes
            src={mapUrl}
            className={cn(
              "w-full h-96 border-0 transition-opacity duration-300",
              mapLoaded ? "opacity-100" : "opacity-0"
            )}
            onLoad={handleIframeLoad}
            onError={handleIframeError}
            title={alt || `Map: ${filename}`}
            sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
            loading="lazy"
            referrerPolicy="strict-origin-when-cross-origin"
          />
          
          {mapLoaded && (
            <div className="absolute top-14 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="flex gap-1">
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => setIsFullscreen(true)}
                  className="h-8 w-8 p-0"
                  title="Fullscreen"
                >
                  <Maximize2 className="h-3 w-3" />
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={handleDownload}
                  className="h-8 w-8 p-0"
                  title="Download HTML"
                >
                  <Download className="h-3 w-3" />
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => window.open(mapUrl, '_blank')}
                  className="h-8 w-8 p-0"
                  title="Open in new tab"
                >
                  <ExternalLink className="h-3 w-3" />
                </Button>
              </div>
            </div>
          )}
        </div>
        
        <div className="p-3 border-t bg-background/50">
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground truncate" title={filename}>
              {filename}
            </p>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setMapLoaded(false)
                setMapError(false)
                // Force iframe reload
                const iframe = document.querySelector(`iframe[src="${mapUrl}"]`) as HTMLIFrameElement
                if (iframe) {
                  iframe.src = iframe.src
                }
              }}
              className="h-6 text-xs opacity-50 hover:opacity-100"
            >
              Reload
            </Button>
          </div>
        </div>
      </Card>

      {/* Fullscreen Modal for Map */}
      {isFullscreen && (
        <div 
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex flex-col"
        >
          <div className="flex items-center justify-between p-4 bg-black/50">
            <div className="flex items-center gap-2 text-white">
              <Map className="h-5 w-5" />
              <span className="font-medium">Interactive Map - {filename}</span>
            </div>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setIsFullscreen(false)}
              className="gap-2"
            >
              <X className="h-4 w-4" />
              Close
            </Button>
          </div>
          <div className="flex-1">
            <iframe
              src={mapUrl}
              className="w-full h-full border-0"
              title={alt || `Fullscreen Map: ${filename}`}
              sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
            />
          </div>
        </div>
      )}
    </>
  )
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
            width={400}
            height={300}
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

      {/* Fullscreen Modal for Image */}
      {isFullscreen && (
        <div 
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4" 
          onClick={() => setIsFullscreen(false)}
        >
          <div className="relative w-full h-full flex items-center justify-center">
            <Image
              src={imageUrl}
              alt={alt || filename}
              fill
              className="object-contain"
              onClick={(e) => e.stopPropagation()}
              sizes="100vw"
              priority
            />
            <Button
              size="sm"
              variant="secondary"
              className="absolute top-4 right-4 z-10"
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

  // Helper function to detect HTML/map files
  const isMapFile = (filename: string) => {
    const isHtml = filename.toLowerCase().endsWith('.html')
    console.log(`File check: ${filename} -> isHtml: ${isHtml}`)
    return isHtml
  }

  // Separate images and maps
  const imageFiles = message.images?.filter(img => !isMapFile(img)) || []
  const mapFiles = message.images?.filter(img => isMapFile(img)) || []
  
  // Debug logging
  console.log('Message images:', message.images)
  console.log('Image files:', imageFiles)
  console.log('Map files:', mapFiles)

  return (
    <div className={cn(
      "flex gap-4 p-6 message-enter border-b border-border/50 last:border-b-0",
      isUser ? "bg-muted/20" : "bg-background",
      "hover:bg-muted/10 transition-colors duration-200"
    )}>
      {/* Avatar */}
      <div className={cn(
        "flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium shadow-sm",
        isUser 
          ? "bg-primary text-primary-foreground" 
          : "bg-gradient-to-br from-blue-500 to-purple-600 text-white"
      )}>
        {isUser ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
      </div>

      {/* Message Content */}
      <div className="flex-1 min-w-0 space-y-3">
        {/* Message Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="font-medium text-foreground">
              {isUser ? 'You' : 'FloatChat Assistant'}
            </span>
            <span className="text-sm text-muted-foreground">
              {formatTime(message.timestamp)}
            </span>
          </div>
          {message.content && (
            <Button
              size="sm"
              variant="ghost"
              onClick={handleCopy}
              className="h-8 w-8 p-0 opacity-60 hover:opacity-100 transition-opacity"
            >
              {copied ? (
                <Check className="h-4 w-4 text-green-600" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </Button>
          )}
        </div>

        {/* Loading State */}
        {message.isLoading && isAssistant && (
          <div className="py-2">
            <TypingIndicator />
          </div>
        )}

        {/* Message Text */}
        {message.content && (
          <div className={cn(
            "prose prose-sm max-w-none dark:prose-invert",
            "prose-headings:text-foreground prose-p:text-foreground",
            "prose-strong:text-foreground prose-code:text-foreground",
            "prose-pre:bg-muted prose-pre:text-foreground"
          )}>
            {isUser ? (
              <p className="text-foreground leading-relaxed">{message.content}</p>
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ node, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '')
                    const inline = !match
                    return !inline && match ? (
                      <CodeBlock language={match[1]}>
                        {String(children).replace(/\n$/, '')}
                      </CodeBlock>
                    ) : (
                      <code className={cn(
                        "bg-muted px-1.5 py-0.5 rounded text-sm font-mono",
                        className
                      )} {...props}>
                        {children}
                      </code>
                    )
                  },
                  p({ children }) {
                    return <p className="leading-relaxed mb-4 last:mb-0">{children}</p>
                  },
                  ul({ children }) {
                    return <ul className="space-y-1 ml-4">{children}</ul>
                  },
                  ol({ children }) {
                    return <ol className="space-y-1 ml-4">{children}</ol>
                  }
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}
          </div>
        )}

        {/* Images */}
        {imageFiles.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
              <ImageIcon className="h-4 w-4" />
              Generated Images ({imageFiles.length})
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {imageFiles.map((image, index) => (
                <ImageDisplay
                  key={index}
                  filename={image}
                  alt={`Generated visualization ${index + 1}`}
                />
              ))}
            </div>
          </div>
        )}

        {/* Maps */}
        {mapFiles.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
              <Map className="h-4 w-4" />
              Interactive Maps ({mapFiles.length})
            </div>
            <div className="space-y-4">
              {mapFiles.map((mapFile, index) => (
                <MapDisplay
                  key={index}
                  filename={mapFile}
                  alt={`Interactive map ${index + 1}`}
                />
              ))}
            </div>
          </div>
        )}

        {/* Agent Steps (Debug Info) */}
        {message.agent_steps && message.agent_steps.length > 0 && (
          <details className="mt-4 group">
            <summary className="text-sm text-muted-foreground cursor-pointer hover:text-foreground transition-colors flex items-center gap-2">
              <span>Show agent execution steps ({message.agent_steps.length})</span>
              <div className="h-px flex-1 bg-border group-open:bg-muted-foreground/30 transition-colors"></div>
            </summary>
            <div className="mt-4 space-y-3">
              {message.agent_steps.map((step, index) => (
                <Card key={index} className="p-4 bg-muted/30 border-l-4 border-l-primary/50">
                  <div className="space-y-3">
                    <div>
                      <div className="text-sm font-medium text-primary mb-2">Action:</div>
                      <pre className="text-sm whitespace-pre-wrap text-muted-foreground bg-background/50 p-3 rounded border">
                        {step.action}
                      </pre>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-primary mb-2">Observation:</div>
                      <pre className="text-sm whitespace-pre-wrap text-muted-foreground bg-background/50 p-3 rounded border">
                        {step.observation}
                      </pre>
                    </div>
                  </div>
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