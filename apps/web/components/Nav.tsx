"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { getToken, setToken } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/calibrate", label: "Calibration game" },
  { href: "/inspect", label: "Inspect a recording" },
];

export function Nav() {
  const pathname = usePathname();
  const [token, setLocalToken] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    // Read the stored token after hydration (localStorage is browser-only).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLocalToken(getToken());
  }, []);

  return (
    <nav className="nav" aria-label="Main">
      <ul>
        {LINKS.map((link) => (
          <li key={link.href}>
            <Link href={link.href} aria-current={pathname === link.href ? "page" : undefined}>
              {link.label}
            </Link>
          </li>
        ))}
      </ul>
      <form
        className="token"
        onSubmit={(event) => {
          event.preventDefault();
          setToken(token.trim());
          setSaved(true);
        }}
      >
        <label htmlFor="api-token">API token</label>
        <input
          id="api-token"
          type="password"
          autoComplete="off"
          placeholder="not needed locally"
          value={token}
          onChange={(event) => {
            setLocalToken(event.target.value);
            setSaved(false);
          }}
        />
        <button type="submit">{saved ? "Saved" : "Save"}</button>
      </form>
    </nav>
  );
}
