// providers/vibeproxy.js
// VibeProxy provider — connects to a local VibeProxy instance that proxies
// requests to various AI providers (Claude, Gemini, GPT) using your existing
// web subscriptions. VibeProxy exposes an OpenAI-compatible API.
//
// NO API KEY NEEDED — VibeProxy handles auth via your browser sessions.
// Glass Turbo auto-detects it on localhost:8317 at startup.

const VIBEPROXY_BASE = 'http://localhost:8317/v1';

/**
 * VibeProxy Provider class — validates connectivity to local VibeProxy.
 */
class VibeProxyProvider {
    /**
     * Validates VibeProxy connectivity by fetching the models list.
     * No API key needed — just checks if the local proxy is reachable.
     * @param {string} _key - Ignored (always 'local')
     * @returns {Promise<{success: boolean, error?: string}>}
     */
    static async validateApiKey(_key) {
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 5000);

            const response = await fetch(`${VIBEPROXY_BASE}/models`, {
                signal: controller.signal,
            });
            clearTimeout(timeout);

            if (response.ok) {
                const data = await response.json();
                const modelCount = data?.data?.length || 0;
                console.log(`[VibeProxy] Connected! Found ${modelCount} models.`);
                return { success: true };
            } else {
                return { success: false, error: `VibeProxy returned status ${response.status}` };
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                return { success: false, error: 'VibeProxy connection timed out. Is it running on port 8317?' };
            }
            console.error('[VibeProxy] Connection error:', error);
            return { success: false, error: `Cannot connect to VibeProxy at localhost:8317. Make sure it is running.` };
        }
    }
}

/**
 * Creates a VibeProxy LLM instance.
 * Uses OpenAI-compatible chat completions API via the local proxy.
 * No Authorization header — VibeProxy handles auth internally.
 */
function createLLM({ apiKey, model, temperature = 0.7, maxTokens = 2048, ...config }) {
    const callApi = async (messages) => {
        const response = await fetch(`${VIBEPROXY_BASE}/chat/completions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: model,
                messages: messages,
                temperature: temperature,
                max_tokens: maxTokens,
            }),
        });

        if (!response.ok) {
            const errText = await response.text().catch(() => '');
            throw new Error(`VibeProxy API error: ${response.status} ${errText}`);
        }

        const result = await response.json();
        return {
            content: result.choices[0].message.content.trim(),
            raw: result,
        };
    };

    return {
        generateContent: async (parts) => {
            const messages = [];
            let systemPrompt = '';
            let userContent = [];

            for (const part of parts) {
                if (typeof part === 'string') {
                    if (systemPrompt === '' && part.includes('You are')) {
                        systemPrompt = part;
                    } else {
                        userContent.push({ type: 'text', text: part });
                    }
                } else if (part.inlineData) {
                    userContent.push({
                        type: 'image_url',
                        image_url: { url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}` },
                    });
                }
            }

            if (systemPrompt) messages.push({ role: 'system', content: systemPrompt });
            if (userContent.length > 0) messages.push({ role: 'user', content: userContent });

            const result = await callApi(messages);
            return {
                response: { text: () => result.content },
                raw: result.raw,
            };
        },

        chat: async (messages) => {
            return await callApi(messages);
        },
    };
}

/**
 * Creates a VibeProxy streaming LLM instance.
 * Uses OpenAI-compatible streaming chat completions via the local proxy.
 * No Authorization header — VibeProxy handles auth internally.
 */
function createStreamingLLM({ apiKey, model, temperature = 0.7, maxTokens = 2048, ...config }) {
    return {
        streamChat: async (messages) => {
            const response = await fetch(`${VIBEPROXY_BASE}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: model,
                    messages: messages,
                    temperature: temperature,
                    max_tokens: maxTokens,
                    stream: true,
                }),
            });

            if (!response.ok) {
                const errText = await response.text().catch(() => '');
                throw new Error(`VibeProxy streaming error: ${response.status} ${errText}`);
            }

            return response;
        },
    };
}

// VibeProxy doesn't support STT
function createSTT(opts) {
    throw new Error('VibeProxy does not support STT. Use Groq or Gemini Live for transcription.');
}

module.exports = {
    VibeProxyProvider,
    createLLM,
    createStreamingLLM,
    createSTT,
};
