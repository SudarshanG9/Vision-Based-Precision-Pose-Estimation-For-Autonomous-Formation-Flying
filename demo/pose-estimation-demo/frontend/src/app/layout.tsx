
import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
    title: "Formation Flight | Pose Estimation Dashboard",
    description:
        "Vision-based precision pose estimation and telemetry " +
        "for autonomous formation-flight research."
};

export default function RootLayout({
    children
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    );
}
