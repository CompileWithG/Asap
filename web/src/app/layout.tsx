import './globals.css'
import { Inter } from 'next/font/google'
import { Toaster } from 'sonner'
import { cn } from '@/lib/utils'
import { ThemeProvider } from '@/components/theme-provider'


const inter = Inter({ 
  subsets: ['latin'],
  display: 'swap',
})

export const metadata = {
  title: 'FloatChat - ARGO Ocean Data Assistant',
  description: 'AI-powered assistant for exploring and analyzing ARGO ocean float data',
  keywords: 'ARGO, ocean data, oceanography, AI assistant, data visualization',
  authors: [{ name: 'FloatChat Team' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#0ea5e9' },
    { media: '(prefers-color-scheme: dark)', color: '#075985' }
  ],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <meta name="robots" content="index, follow" />
      </head>
      <body className={cn(
        inter.className,
        "min-h-screen bg-background font-sans antialiased"
      )}>
        
        <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            disableTransitionOnChange
          >
        <div className="relative flex min-h-screen flex-col">
          {children}
        </div>
        <Toaster 
          position="top-right"
          richColors
          expand
          closeButton
        />
        
        </ThemeProvider>
       
      </body>
    </html>
  )
}