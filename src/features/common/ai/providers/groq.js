// providers/groq.js
// Groq provider — uses Whisper v3 Turbo for speech-to-text via chunked REST API calls.
// Uses voice-activity-based chunking: accumulates audio while speech is detected,
// flushes when silence is detected for ~500ms, with a max window of 10 seconds.

const GROQ_API_BASE = 'https://api.groq.com/openai/v1';

/**
 * Groq Provider class — handles API key validation.
 */
class GroqProvider {
    /**
     * Validates a Groq API key by listing models.
     * @param {string} key
     * @returns {Promise<{success: boolean, error?: string}>}
     */
    static async validateApiKey(key) {
        if (!key || typeof key !== 'string') {
            return { success: false, error: 'Invalid Groq API key format.' };
        }
        try {
            const response = await fetch(`${GROQ_API_BASE}/models`, {
                headers: { 'Authorization': `Bearer ${key}` }
            });

            if (response.ok) {
                return { success: true };
            } else {
                const errorData = await response.json().catch(() => ({}));
                const message = errorData.error?.message || `Validation failed with status: ${response.status}`;
                return { success: false, error: message };
            }
        } catch (error) {
            console.error('[GroqProvider] Network error during key validation:', error);
            return { success: false, error: 'A network error occurred during validation.' };
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// WAV encoding helpers
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Wraps raw PCM-16 mono data in a minimal WAV header.
 * @param {Buffer} pcmBuffer - Raw PCM-16 LE mono audio
 * @param {number} sampleRate
 * @returns {Buffer}
 */
function pcmToWav(pcmBuffer, sampleRate = 24000) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataSize = pcmBuffer.length;
    const headerSize = 44;

    const wav = Buffer.alloc(headerSize + dataSize);

    // RIFF header
    wav.write('RIFF', 0);
    wav.writeUInt32LE(36 + dataSize, 4);     // file size - 8
    wav.write('WAVE', 8);

    // fmt  sub-chunk
    wav.write('fmt ', 12);
    wav.writeUInt32LE(16, 16);               // sub-chunk size
    wav.writeUInt16LE(1, 20);                // audio format (PCM)
    wav.writeUInt16LE(numChannels, 22);
    wav.writeUInt32LE(sampleRate, 24);
    wav.writeUInt32LE(byteRate, 28);
    wav.writeUInt16LE(blockAlign, 32);
    wav.writeUInt16LE(bitsPerSample, 34);

    // data sub-chunk
    wav.write('data', 36);
    wav.writeUInt32LE(dataSize, 40);
    pcmBuffer.copy(wav, headerSize);

    return wav;
}

// ──────────────────────────────────────────────────────────────────────────────
// Audio analysis helpers
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Calculate the RMS (Root Mean Square) energy of a PCM-16 LE buffer.
 * Returns a value between 0 (silence) and 1 (max volume).
 * @param {Buffer} pcmBuffer
 * @returns {number}
 */
function calculateRMS(pcmBuffer) {
    if (pcmBuffer.length < 2) return 0;

    const sampleCount = Math.floor(pcmBuffer.length / 2);
    let sumSquares = 0;

    for (let i = 0; i < sampleCount; i++) {
        const sample = pcmBuffer.readInt16LE(i * 2);
        sumSquares += sample * sample;
    }

    const rms = Math.sqrt(sumSquares / sampleCount);
    return rms / 32768; // Normalize to 0..1
}

// Minimum RMS energy to consider audio as "speech" (not silence).
// Typical background noise is ~0.005-0.01; speech is usually > 0.02.
const SILENCE_RMS_THRESHOLD = 0.012;

// How many consecutive "silent" analysis frames before we consider it a silence gap
const SILENCE_FRAME_SIZE_MS = 100;  // Analyze in 100ms frames

/**
 * Known Whisper hallucination phrases that appear on silent/near-silent audio.
 * These are compared in lowercase after trimming.
 */
const HALLUCINATION_PHRASES = new Set([
    'thank you',
    'thank you.',
    'thanks.',
    'thanks',
    'thank you for watching',
    'thank you for watching.',
    'thanks for watching.',
    'thanks for watching',
    'thank you so much.',
    'thank you so much',
    'bye.',
    'bye',
    'goodbye.',
    'goodbye',
    'you',
    'the end.',
    'the end',
    'subtitles by the amara.org community',
    'subs by www.telesynced.com',
    '...',
    '',
    // Turkish hallucinations
    'teşekkür ederim.',
    'teşekkür ederim',
    'teşekkürler.',
    'teşekkürler',
    'altyazı',
    'abone ol',
]);

/**
 * Checks if a transcript looks like a Whisper hallucination.
 * @param {string} text
 * @returns {boolean}
 */
function isHallucination(text) {
    if (!text) return true;
    const cleaned = text.trim().toLowerCase();
    if (cleaned.length === 0) return true;
    if (HALLUCINATION_PHRASES.has(cleaned)) return true;
    // Catch repeated single words like "Thank you. Thank you. Thank you."
    const words = cleaned.replace(/[.,!?]/g, '').split(/\s+/);
    if (words.length >= 2) {
        const unique = new Set(words);
        if (unique.size === 1) return true; // All same word repeated
        if (unique.size <= 2 && words.length >= 4) return true; // e.g. "thank you thank you"
    }
    return false;
}

// ──────────────────────────────────────────────────────────────────────────────
// STT session (silence-based chunking)
// ──────────────────────────────────────────────────────────────────────────────

const SAMPLE_RATE = 24000;
const BYTES_PER_SAMPLE = 2; // PCM-16

// Silence-based chunking parameters
const SILENCE_DURATION_MS = 500;       // Flush after 500ms of silence (natural pause)
const MAX_CHUNK_DURATION_S = 10;       // Max window: force flush after 10 seconds
const MIN_SPEECH_DURATION_S = 0.5;     // Minimum audio to even bother sending

const MAX_CHUNK_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * MAX_CHUNK_DURATION_S;
const MIN_SPEECH_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * MIN_SPEECH_DURATION_S;
const SILENCE_FRAME_BYTES = Math.floor(SAMPLE_RATE * BYTES_PER_SAMPLE * (SILENCE_FRAME_SIZE_MS / 1000));

/**
 * Creates a Groq STT session that buffers PCM audio and uses voice-activity
 * detection to find natural speech boundaries before calling the Groq Whisper
 * transcription endpoint.
 *
 * The returned object exposes the same interface as other STT providers:
 *   - sendRealtimeInput(base64Audio)
 *   - close()
 *
 * Language is NOT specified so Whisper auto-detects it.
 * This allows seamless Turkish + English transcription.
 *
 * @param {object} opts
 * @param {string} opts.apiKey
 * @param {object} [opts.callbacks]
 * @returns {object}
 */
function createSTT({ apiKey, callbacks = {}, ...config }) {
    let audioBuffer = Buffer.alloc(0);
    let closed = false;
    let pendingRequest = false;

    // Voice activity tracking
    let isSpeaking = false;
    let silenceStartTime = null;
    let speechStartTime = null;

    // Safety timer: check for max duration and silence gaps periodically
    let checkTimer = null;

    /**
     * Flush the accumulated audio buffer by sending it to Groq.
     */
    async function flush() {
        if (closed || audioBuffer.length === 0 || pendingRequest) return;

        const pcmData = audioBuffer;
        audioBuffer = Buffer.alloc(0);

        // Reset voice activity state
        isSpeaking = false;
        silenceStartTime = null;
        speechStartTime = null;

        // Skip very short audio chunks
        if (pcmData.length < MIN_SPEECH_BYTES) return;

        // Final RMS check on the entire chunk
        const rms = calculateRMS(pcmData);
        if (rms < SILENCE_RMS_THRESHOLD) {
            return; // Entire chunk is silence
        }

        pendingRequest = true;
        try {
            const wavBuffer = pcmToWav(pcmData, SAMPLE_RATE);

            // Build multipart/form-data
            const blob = new Blob([wavBuffer], { type: 'audio/wav' });
            const form = new FormData();
            form.append('file', blob, 'audio.wav');
            form.append('model', 'whisper-large-v3-turbo');
            form.append('response_format', 'json');
            // Prompt hint biases Whisper toward Turkish + English and away from
            // CJK/Southeast-Asian false detections on ambiguous audio.
            form.append('prompt', 'Bu bir Türkçe ve İngilizce konuşma. This is a Turkish and English conversation.');

            const response = await fetch(`${GROQ_API_BASE}/audio/transcriptions`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                },
                body: form,
            });

            if (!response.ok) {
                const errBody = await response.text().catch(() => '');
                console.error(`[Groq STT] Transcription failed: ${response.status} ${errBody}`);
                return;
            }

            const result = await response.json();
            const text = result.text?.trim();

            // ── Language script filter ──
            // Reject text that contains CJK, Japanese, Korean, Thai, or Vietnamese
            // tonal marks — these are guaranteed false detections for a TR+EN speaker.
            const NON_LATIN_SCRIPT_REGEX = /[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u0E00-\u0E7F\u1E00-\u1EFF\u0300-\u036F]{3,}/;
            if (text && NON_LATIN_SCRIPT_REGEX.test(text)) {
                console.log(`[Groq STT] Filtered non-TR/EN script: "${text.substring(0, 60)}..."`);
                return;
            }

            // ── Hallucination filter ──
            if (text && text.length > 0 && !isHallucination(text)) {
                // Emit in the same shape that the Whisper (local) provider uses
                // so that sttService.js can handle it with the existing code path.
                callbacks.onmessage?.({
                    provider: 'groq',
                    text: text,
                });
            } else if (text) {
                console.log(`[Groq STT] Filtered hallucination: "${text}"`);
            }
        } catch (error) {
            console.error('[Groq STT] Error during transcription:', error);
            callbacks.onerror?.(error);
        } finally {
            pendingRequest = false;
        }
    }

    /**
     * Analyze incoming audio chunk for voice activity.
     * Called each time new audio data arrives.
     */
    function processVoiceActivity(pcmChunk) {
        const rms = calculateRMS(pcmChunk);
        const now = Date.now();
        const hasSpeech = rms >= SILENCE_RMS_THRESHOLD;

        if (hasSpeech) {
            // Speech detected
            if (!isSpeaking) {
                isSpeaking = true;
                speechStartTime = speechStartTime || now;
            }
            silenceStartTime = null; // Reset silence counter
        } else {
            // Silence detected
            if (isSpeaking && !silenceStartTime) {
                silenceStartTime = now;
            }
        }
    }

    /**
     * Periodic check: flush on silence gap or max duration
     */
    function periodicCheck() {
        if (closed || audioBuffer.length === 0) return;

        const now = Date.now();

        // Force flush if we've accumulated max duration
        if (audioBuffer.length >= MAX_CHUNK_BYTES) {
            console.log('[Groq STT] Max chunk duration reached, flushing...');
            flush();
            return;
        }

        // Flush on silence gap (only if we have speech content)
        if (isSpeaking && silenceStartTime && (now - silenceStartTime) >= SILENCE_DURATION_MS) {
            if (audioBuffer.length >= MIN_SPEECH_BYTES) {
                console.log(`[Groq STT] Silence gap detected (${now - silenceStartTime}ms), flushing...`);
                flush();
            }
        }
    }

    // Check every 100ms for silence gaps and max duration
    checkTimer = setInterval(periodicCheck, 100);

    return {
        /**
         * Accepts base64-encoded PCM-16 mono audio.
         * @param {string} base64Audio
         */
        sendRealtimeInput: (base64Audio) => {
            if (closed) return;

            const pcmChunk = Buffer.from(base64Audio, 'base64');
            audioBuffer = Buffer.concat([audioBuffer, pcmChunk]);

            // Analyze this chunk for voice activity
            processVoiceActivity(pcmChunk);
        },

        close: async () => {
            closed = true;
            if (checkTimer) {
                clearInterval(checkTimer);
                checkTimer = null;
            }
            // Final flush of remaining audio
            await flush();
            console.log('[Groq STT] Session closed.');
        },
    };
}

// Groq does not provide LLM or streaming LLM through this provider
function createLLM(opts) {
    console.warn('[Groq] LLM not supported via this provider.');
    return {
        generateContent: async () => { throw new Error('Groq does not support LLM functionality in Glass.'); }
    };
}

function createStreamingLLM(opts) {
    console.warn('[Groq] Streaming LLM not supported via this provider.');
    return {
        streamChat: async () => { throw new Error('Groq does not support Streaming LLM functionality in Glass.'); }
    };
}

module.exports = {
    GroqProvider,
    createSTT,
    createLLM,
    createStreamingLLM
};
