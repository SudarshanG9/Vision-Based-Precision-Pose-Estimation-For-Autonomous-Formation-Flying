
"use client";

import {
    Pause,
    Play,
    RotateCcw,
    SkipBack
} from "lucide-react";

interface Props {
    frameIndex: number;
    frameCount: number;
    playing: boolean;
    playbackFps: number;
    disabled?: boolean;
    onToggle: () => void;
    onStop: () => void;
    onSeek: (index: number) => void;
    onPlaybackFpsChange: (fps: number) => void;
}

export default function PlaybackControls({
    frameIndex,
    frameCount,
    playing,
    playbackFps,
    disabled = false,
    onToggle,
    onStop,
    onSeek,
    onPlaybackFpsChange
}: Props) {
    const lastFrame = Math.max(0, frameCount - 1);

    return (
        <div className="playback-controls">
            <div className="playback-time-row">
                <span>Timeline</span>
                <strong>
                    {frameCount > 0 ? frameIndex + 1 : 0}
                    {" / "}
                    {frameCount}
                </strong>
            </div>

            <input
                type="range"
                min={0}
                max={lastFrame}
                step={1}
                value={Math.min(frameIndex, lastFrame)}
                disabled={disabled || frameCount === 0}
                onChange={(event) =>
                    onSeek(Number(event.target.value))
                }
                aria-label="Playback timeline"
            />

            <div className="playback-buttons">
                <button
                    type="button"
                    className="icon-button"
                    onClick={() => onSeek(0)}
                    disabled={disabled || frameCount === 0}
                    title="Beginning"
                >
                    <SkipBack size={16} />
                </button>

                <button
                    type="button"
                    className="play-button"
                    onClick={onToggle}
                    disabled={disabled || frameCount <= 1}
                >
                    {playing ? <Pause size={16} /> : <Play size={16} />}
                    {playing ? "Pause" : "Play"}
                </button>

                <button
                    type="button"
                    className="icon-button"
                    onClick={onStop}
                    disabled={disabled || frameCount === 0}
                    title="Stop and rewind"
                >
                    <RotateCcw size={16} />
                </button>

                <select
                    className="playback-speed"
                    value={playbackFps}
                    onChange={(event) =>
                        onPlaybackFpsChange(Number(event.target.value))
                    }
                    aria-label="Playback frames per second"
                >
                    <option value={10}>10 fps</option>
                    <option value={20}>20 fps</option>
                    <option value={30}>30 fps</option>
                    <option value={60}>60 fps</option>
                </select>
            </div>
        </div>
    );
}
