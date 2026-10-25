
"use client";

import {
    useCallback,
    useEffect,
    useRef,
    useState
} from "react";

interface PlaybackState {
    frameIndex: number;
    playing: boolean;
    playbackFps: number;
    setPlaybackFps: (fps: number) => void;
    seek: (index: number) => void;
    stop: () => void;
    togglePlaying: () => void;
}

export function usePlayback(
    frameCount: number,
    sourceFps: number,
    initialPlaybackFps = 30
): PlaybackState {
    const [frameIndex, setFrameIndex] = useState(0);
    const [playing, setPlaying] = useState(false);

    const [playbackFps, setPlaybackFps] = useState(
        initialPlaybackFps
    );

    const frameRef = useRef(0);
    const anchorRef = useRef<{
        wallTime: number;
        simulationFrame: number;
    } | null>(null);

    const maxIndex = Math.max(0, frameCount - 1);

    const seek = useCallback(
        (index: number) => {
            const next = Math.max(
                0,
                Math.min(maxIndex, Math.round(index))
            );

            frameRef.current = next;
            anchorRef.current = null;
            setFrameIndex(next);
        },
        [maxIndex]
    );

    const stop = useCallback(() => {
        frameRef.current = 0;
        anchorRef.current = null;
        setFrameIndex(0);
        setPlaying(false);
    }, []);

    const togglePlaying = useCallback(() => {
        if (frameCount <= 1) return;

        if (!playing && frameRef.current >= maxIndex) {
            frameRef.current = 0;
            setFrameIndex(0);
        }

        anchorRef.current = null;
        setPlaying((current) => !current);
    }, [frameCount, maxIndex, playing]);

    useEffect(() => {
        stop();
    }, [frameCount, stop]);

    useEffect(() => {
        if (!playing || frameCount <= 1) return;

        const simulationRate = Math.max(1, sourceFps);
        const presentationRate = Math.max(1, playbackFps);

        let animationId = 0;

        function update(now: number) {
            if (anchorRef.current === null) {
                anchorRef.current = {
                    wallTime: now,
                    simulationFrame: frameRef.current
                };
            }

            const anchor = anchorRef.current;

            const elapsedSeconds = (
                now - anchor.wallTime
            ) / 1000;

            // One second of real time corresponds to one second
            // of recorded simulation time.
            const next = Math.min(
                maxIndex,
                Math.floor(
                    anchor.simulationFrame +
                    elapsedSeconds * simulationRate
                )
            );

            if (next !== frameRef.current) {
                frameRef.current = next;
                setFrameIndex(next);
            }

            if (next >= maxIndex) {
                setPlaying(false);
                anchorRef.current = null;
                return;
            }

            animationId = requestAnimationFrame(update);
        }

        // Cap React state updates to the selected presentation rate
        // while preserving simulation-time frame selection.
        const interval = window.setInterval(() => {
            if (animationId !== 0) {
                cancelAnimationFrame(animationId);
            }

            animationId = requestAnimationFrame(update);
        }, 1000 / presentationRate);

        return () => {
            window.clearInterval(interval);
            cancelAnimationFrame(animationId);
            anchorRef.current = null;
        };
    }, [
        playing,
        frameCount,
        sourceFps,
        playbackFps,
        maxIndex
    ]);

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
