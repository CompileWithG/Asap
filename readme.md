# FloatChat 🌊 - Next.js Frontend

A modern, responsive Next.js frontend for the ARGO Ocean Data Assistant, providing real-time chat interface with advanced data visualization capabilities.

## 🚀 Features

- **Real-time Chat Interface** - WebSocket integration with fallback to HTTP
- **Advanced Data Visualization** - Display generated ocean data plots and maps
- **Session Management** - Persistent chat history with local storage
- **Responsive Design** - Mobile-first approach with Tailwind CSS
- **Dark/Light Mode** - System preference detection with manual toggle
- **Type Safety** - Full TypeScript implementation
- **Modern UI** - Clean, ocean-themed interface with smooth animations

## 📁 Project Structure

```
floatchat-frontend/
├── app/
│   ├── globals.css          # Global styles and Tailwind
│   ├── layout.tsx           # Root layout with providers
│   └── page.tsx             # Main chat page
├── components/
│   ├── ui/                  # Reusable UI components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── input.tsx
│   │   └── textarea.tsx
│   ├── ChatInput.tsx        # Message input component
│   ├── ChatMessage.tsx      # Message display component
│   └── Sidebar.tsx          # Chat history sidebar
├── lib/
│   ├── api.ts              # API client and WebSocket manager
│   ├── store.ts            # Zustand state management
│   └── utils.ts            # Utility functions
├── package.json
├── tailwind.config.js
├── next.config.js
└── tsconfig.json
```

## 🛠️ Installation & Setup

### Prerequisites

- Node.js 18+ and npm/yarn
- Running FastAPI backend on `http://localhost:8000`

### 1. Create Next.js Project

```bash
# Create new Next.js app
npx create-next-app@latest floatchat-frontend --typescript --tailwind --app

# Navigate to project directory
cd floatchat-frontend
```

### 2. Install Dependencies

```bash
# Install all required packages
npm install next react react-dom @types/node @types/react @types/react-dom typescript tailwindcss autoprefixer postcss class-variance-authority clsx tailwind-merge lucide-react axios zustand react-markdown remark-gfm react-syntax-highlighter @types/react-syntax-highlighter sonner
```

### 3. Environment Configuration

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Replace Default Files

Replace the contents of these files with the provided code:

- `app/globals.css` - Custom styles and animations
- `app/layout.tsx` - Root layout with metadata
- `app/page.tsx` - Main chat interface
- `tailwind.config.js` - Custom theme configuration
- `next.config.js` - Next.js configuration
- `tsconfig.json` - TypeScript configuration

### 5. Create Component Files

Create the component structure and copy the provided code:

```bash
mkdir -p components/ui lib
```

- `components/ui/button.tsx`
- `components/ui/card.tsx`
- `components/ui/input.tsx`
- `components/ui/textarea.tsx`
- `components/ChatInput.tsx`
- `components/ChatMessage.tsx`
- `components/Sidebar.tsx`
- `lib/api.ts`
- `lib/store.ts`
- `lib/utils.ts`

## 🚦 Running the Application

### Development Mode

```bash
npm run dev
# or
yarn dev
```

Visit `http://localhost:3000` to access the application.

### Production Build

```bash
npm run build
npm start
```

## 🔧 Configuration Options

### Environment Variables

- `NEXT_PUBLIC_API_URL` - FastAPI backend URL (default: http://localhost:8000)
- `NEXT_PUBLIC_WS_URL` - WebSocket URL (auto-derived from API_URL)

### API Integration

The frontend automatically handles:

- **HTTP Fallback** - Uses REST API if WebSocket fails
- **Connection Monitoring** - Shows connection status
- **Error Handling** - User-friendly error messages
- **Image Loading** - Displays generated plots with loading states

### State Management

Uses Zustand for:

- **Chat History** - Persistent across sessions
- **UI State** - Sidebar, dark mode, etc.
- **Connection Status** - API and WebSocket states

## 🎨 Customization

### Theme Colors

Edit `tailwind.config.js` to customize the ocean theme:

```javascript
colors: {
  ocean: {
    50: '#f0f9ff',
    500: '#0ea5e9',
    900: '#0c4a6e',
  }
}
```

### UI Components

All components are modular and customizable:

- Modify `components/ui/` for base components
- Update `components/ChatMessage.tsx` for message styling
- Customize `components/Sidebar.tsx` for navigation

## 🧪 Testing Chat Interface

### Example Queries to Test

1. **Simple Data Query:**
   ```
   Show me temperature data from Bay of Bengal
   ```

2. **With Visualizations:**
   ```
   Create scatter plots for temperature vs salinity in the North Atlantic
   ```

3. **BGC Parameters:**
   ```
   Generate a global map of dissolved oxygen (DOXY) data quality
   ```

4. **Specific Regions:**
   ```
   Find ARGO floats between latitude 10-20 and longitude 80-95 with temperature above 25°C
   ```

## 📱 Mobile Responsiveness

The interface is fully responsive with:

- **Mobile-first Design** - Optimized for touch interactions
- **Collapsible Sidebar** - Overlay mode on smaller screens
- **Touch-friendly Controls** - Larger touch targets
- **Responsive Images** - Optimized loading for mobile networks

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure FastAPI backend is running on port 8000
   - Check CORS settings in FastAPI
   - Verify NEXT_PUBLIC_API_URL in .env.local

2. **WebSocket Connection Issues**
   - WebSocket will auto-fallback to HTTP
   - Check browser console for connection errors
   - Ensure no proxy/firewall blocking WebSocket

3. **Images Not Loading**
   - Verify image URLs in browser network tab
   - Check FastAPI static file serving
   - Ensure proper CORS headers for images

4. **Build Errors**
   - Clear .next folder: `rm -rf .next`
   - Reinstall dependencies: `rm -rf node_modules && npm install`
   - Check TypeScript errors: `npx tsc --noEmit`

### Debug Mode

Enable debug logging by adding to .env.local:
```env
NEXT_PUBLIC_DEBUG=true
```

## 🚀 Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set environment variables in Vercel dashboard
# NEXT_PUBLIC_API_URL=https://your-api-domain.com
```

### Docker Deployment

```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
```

### Static Export

For static hosting:

```javascript
// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  }
}
```

## 🛡️ Security Considerations

- **API Keys** - Never expose backend API keys in frontend
- **CORS** - Configure proper CORS policies in FastAPI
- **CSP** - Content Security Policy headers configured
- **XSS Protection** - React's built-in XSS protection
- **Input Validation** - All user inputs are sanitized

## 📊 Performance Optimization

- **Code Splitting** - Automatic route-based splitting
- **Image Optimization** - Next.js Image component
- **Bundle Analysis** - Use `npm run analyze`
- **Caching** - API responses cached where appropriate

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test thoroughly
4. Commit: `git commit -m 'Add new feature'`
5. Push: `git push origin feature/new-feature`
6. Submit pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

- **Issues** - Report bugs via GitHub Issues
- **Discussions** - Feature requests and questions
- **Documentation** - Check inline code comments

---

## 🏁 Quick Start Checklist

- [ ] FastAPI backend running on port 8000
- [ ] Node.js 18+ installed
- [ ] Created Next.js project
- [ ] Installed all dependencies
- [ ] Created .env.local with API URL
- [ ] Copied all component files
- [ ] Started development server
- [ ] Tested chat functionality
- [ ] Verified image loading
- [ ] Checked mobile responsiveness

Your FloatChat frontend should now be running at `http://localhost:3000` with full integration to your FastAPI backend!