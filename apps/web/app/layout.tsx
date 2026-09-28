import type { Metadata } from "next";

import { Nav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "neurolayer",
  description: "Neural intelligence layer: from neural signals to computer interactions.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <Nav />
        {children}
      </body>
    </html>
  );
}
