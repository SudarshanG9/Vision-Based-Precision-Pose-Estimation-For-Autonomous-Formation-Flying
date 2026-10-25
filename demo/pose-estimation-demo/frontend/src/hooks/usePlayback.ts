
"use client";

import { useCallback, useEffect, useState } from "react";

export function usePlayback(
    frameCount: number,
    sourceFps: number,
    initialPlaybackFps = 30
) {
    const [frameIndex, setFrameIndex] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [playbackFps, setPlaybackFps] = useState(
        initialPlaybackFps
    );

    const safeCount = Math.max(0, frameCount);

    const seek = useCallback(
        (index: number) => {
            const last = Math.max(0, safeCount - 1);

            setFrameIndex(
                Math.max(0, Math.min(last, Math.round(index)))
            );
        },
        [safeCount]
    );

    const stop = useCallback(() => {
        setPlaying(false);
        setFrameIndex(0);
    }, []);

    useEffect(() => {
        setPlaying(false);
        setFrameIndex(0);
    }, [frameCount]);

    useEffect(() => {
        if (!playing || safeCount <= 1) return;

        const safeSourceFps = Math.max(1, sourceFps);
        const safePlaybackFps = Math.max(1, playbackFps);

        const intervalMs = 1000 / safePlaybackFps;

        const timer = window.setInterval(() => {
            setFrameIndex((current) => {
                if (current >= safeCount - 1) {
                    setPlaying(false);
                    return current;
                }

                // Playback advances one available telemetry frame.
                // sourceFps is retained as a separate simulation property.
                void safeSourceFps;

                return current + 1;
            });
        }, intervalMs);

        return () => window.clearInterval(timer);
    }, [playing, safeCount, sourceFps, playbackFps]);

    const togglePlaying = useCallback(() => {
        if (safeCount <= 1) return;

        setPlaying((current) => !current);

        if (frameIndex >= safeCount - 1) {
            setFrameIndex(0);
        }
    }, [safeCount, frameIndex]);

    return {
        frameIndex,
        playing,
        playbackFps,
        setPlaybackFps,
        seek,
        stop,
        togglePlaying
    };
}
