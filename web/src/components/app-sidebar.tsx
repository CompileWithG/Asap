"use client"

import * as React from "react"
import {
  IconBrandGithub,
  IconBrandTwitter,
  IconChartAreaLine,
  IconCompass,
  IconFileText,
  IconHelp,
  IconSearch,
  IconSettings,
  IconShip,
  IconWorld,
  IconDashboard
} from "@tabler/icons-react"

import { NavDocuments } from "@/components/nav-documents"
import { NavMain } from "@/components/nav-main"
import { NavSecondary } from "@/components/nav-secondary"
import { NavUser } from "@/components/nav-user"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"

const data = {
  user: {
    name: "ARGO User",
    email: "user@example.com",
    avatar: "/avatars/argo-user.png", // Assuming you have an ARGO-themed avatar
  },
  navMain: [
    {
      title: "New Chat",
      url: "/",
      icon: IconFileText,
    },
    {
      title: "Dashboard",
      url: "/",
      icon: IconDashboard,
    },
    {
      title: "Explore Data",
      url: "#",
      icon: IconWorld,
    },
    {
      title: "Visualizations",
      url: "#",
      icon: IconChartAreaLine,
    },
    {
      title: "Float Tracking",
      url: "#",
      icon: IconCompass,
    },
    {
      title: "ARGO Floats",
      url: "#",
      icon: IconShip,
    },
  ],
  navSecondary: [
    {
      title: "Settings",
      url: "#",
      icon: IconSettings,
    },
    {
      title: "GitHub",
      url: "https://github.com/CompileWithG/Asap", // Replace with your GitHub URL
      icon: IconBrandGithub,
    },
    {
      title: "Twitter",
      url: "#",
      icon: IconHelp,
    },
    {
      title: "Search",
      url: "#",
      icon: IconSearch,
    },
  ],
  documents: [
    {
      name: "About ARGO",
      url: "#", 
      icon: IconFileText,
    },
    {
      name: "Documentation",
      url: "#", // Link to documentation
      icon: IconFileText,
    }
  ],
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              className="data-[slot=sidebar-menu-button]:!p-1.5"
            > 
              <a href="/">
                <IconShip className="!size-5" /> {/* Changed icon to reflect ARGO theme */}
                <span className="text-base font-semibold">FloatChat</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
        <NavDocuments items={data.documents} />
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter>
    </Sidebar>
  )
}
