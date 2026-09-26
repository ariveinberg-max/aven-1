import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "neurolayer",
  description: "Neural intelligence layer: from neural signals to computer interactions.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
