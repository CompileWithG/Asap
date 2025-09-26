/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {},
  },
  images: {
    domains: ['localhost', '127.0.0.1'],
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/api/images/**',
      }
    ],
  },
  webpack: (config: any, { isServer }: { isServer: boolean }) => {
    // Handle canvas for node environment (for image processing)
    if (isServer) {
      config.externals.push({
        canvas: 'canvas',
      })
    }
    return config
  },
  // Enable source maps in production for better debugging
  productionBrowserSourceMaps: true,
  
  // Security headers
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
        ],
      },
    ]
  },
}

module.exports = nextConfig