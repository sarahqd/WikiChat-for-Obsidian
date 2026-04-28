/**
 * Unified LLM Client
 * Manages multiple LLM providers and model configurations
 */

import type { 
    LLMWikiSettings, 
    LLMProvider, 
    ModelConfig, 
    ProviderConfig
} from '../types';
import { getProviderMetadata } from '../types';
import type { LLMProviderInterface, LLMProviderConfig, LLMChatOptions, LLMStreamOptions, LLMResponse } from './types';
import { OllamaClient } from '../ollama/client';
import { OpenAIProvider } from './openai';
import { AnthropicProvider } from './anthropic';

export class LLMClient implements LLMProviderInterface {
    readonly name = 'unified';
    private settings: LLMWikiSettings;
    private providers: Map<string, LLMProviderInterface> = new Map();
    private currentModel: ModelConfig | null = null;

    constructor(settings: LLMWikiSettings) {
        this.settings = settings;
        this.initializeProviders();
        this.setCurrentModel(settings.currentModelId);
    }

    private getModelForProvider(provider: LLMProvider): ModelConfig | undefined {
        return this.settings.models.find((model) => model.provider === provider);
    }

    private initializeProviders(): void {
        for (const config of this.settings.providers) {
            const model = this.getModelForProvider(config.name);
            this.initializeProviderFromConfig(config, model);
        }
    }

    private wrapOllamaClient(providerName: LLMProvider, client: OllamaClient): LLMProviderInterface {
        return {
            name: providerName,
            setModel: (model: string) => {
                client.setModel(model);
            },
            chat: async (options: LLMChatOptions): Promise<LLMResponse> => {
                const message = await client.chat(
                    options.messages,
                    options.tools,
                    options.systemPrompt
                );
                return { message };
            },
            chatStream: async (options: LLMStreamOptions): Promise<LLMResponse> => {
                const message = await client.chatStream(
                    options.messages,
                    options.onChunk,
                    options.tools,
                    options.systemPrompt,
                    options.signal  // Pass abort signal
                );
                return { message };
            },
            embed: async (text: string): Promise<number[]> => {
                return client.embed(text);
            },
            listModels: async (): Promise<string[]> => {
                return client.listModels();
            },
            healthCheck: async (): Promise<boolean> => {
                return client.healthCheck();
            },
        };
    }

    private getProvider(name: LLMProvider): LLMProviderInterface | undefined {
        return this.providers.get(name);
    }

    setCurrentModel(modelId: string): void {
        const model = this.settings.models.find(m => m.id === modelId);
        if (model) {
            this.currentModel = model;
        } else if (this.settings.models.length > 0) {
            // Fallback to first model or default
            this.currentModel = this.settings.models.find(m => m.isDefault) || this.settings.models[0];
        } else {
            // No available models
            this.currentModel = null;
        }
    }

    getCurrentModel(): ModelConfig | null {
        return this.currentModel;
    }

    updateSettings(settings: LLMWikiSettings): void {
        this.settings = settings;
        this.providers.clear();
        this.initializeProviders();
        this.setCurrentModel(settings.currentModelId);
    }

    private getProviderForModel(model: ModelConfig): LLMProviderInterface {
        const config = this.settings.providers.find(p => p.name === model.provider);
        if (config) {
            this.initializeProviderFromConfig(config, model);
        }

        const provider = this.getProvider(model.provider);

        if (!provider) {
            throw new Error(`Provider ${model.provider} is not available or not configured`);
        }

        return provider;
    }

    private buildRuntimeConfig(config: ProviderConfig, model?: ModelConfig): LLMProviderConfig {
        const metadata = getProviderMetadata(config.name);

        return {
            apiKey: model?.apiKey || config.apiKey,
            baseUrl: model?.baseUrl || config.baseUrl || metadata.defaultBaseUrl,
            model: model?.modelId || metadata.defaultModelId,
            defaultModel: metadata.defaultModelId,
        };
    }

    private canInitializeProvider(config: ProviderConfig, model: ModelConfig | undefined, runtimeConfig: LLMProviderConfig): boolean {
        const metadata = getProviderMetadata(config.name);
        const hasBaseUrl = Boolean(runtimeConfig.baseUrl);
        const hasApiKey = Boolean(runtimeConfig.apiKey);
        const modelHasConfig = Boolean(model?.baseUrl);

        if (!hasBaseUrl) {
            return false;
        }

        if (metadata.authMode === 'required' && !hasApiKey) {
            return false;
        }

        return config.enabled || modelHasConfig;
    }

    private createProviderInstance(providerName: LLMProvider, runtimeConfig: LLMProviderConfig): LLMProviderInterface {
        const metadata = getProviderMetadata(providerName);

        switch (metadata.apiStyle) {
            case 'ollama': {
                const client = new OllamaClient(runtimeConfig.baseUrl || metadata.defaultBaseUrl, runtimeConfig.model || metadata.defaultModelId);
                return this.wrapOllamaClient(providerName, client);
            }
            case 'anthropic':
                return new AnthropicProvider(runtimeConfig);
            case 'openai':
            default:
                return new OpenAIProvider(runtimeConfig);
        }
    }

    private initializeProviderFromConfig(config: ProviderConfig, model?: ModelConfig): void {
        const runtimeConfig = this.buildRuntimeConfig(config, model);
        if (!this.canInitializeProvider(config, model, runtimeConfig)) {
            return;
        }

        this.providers.set(config.name, this.createProviderInstance(config.name, runtimeConfig));
    }

    async chat(options: LLMChatOptions): Promise<LLMResponse> {
        if (!this.currentModel) {
            throw new Error('No model selected');
        }

        const provider = this.getProviderForModel(this.currentModel);
        
        if (provider.setModel) {
            provider.setModel(this.currentModel.modelId);
        }

        return provider.chat(options);
    }

    async chatStream(options: LLMStreamOptions): Promise<LLMResponse> {
        if (!this.currentModel) {
            throw new Error('No model selected');
        }

        const provider = this.getProviderForModel(this.currentModel);
        
        if (provider.setModel) {
            provider.setModel(this.currentModel.modelId);
        }

        return provider.chatStream(options);
    }

    async embed(text: string): Promise<number[]> {
        if (!this.currentModel) {
            throw new Error('No model selected');
        }

        const provider = this.getProviderForModel(this.currentModel);
        if (!provider.embed) {
            throw new Error(`Provider ${this.currentModel.provider} does not support embeddings`);
        }

        return provider.embed(text);
    }

    async listModels(): Promise<string[]> {
        // Return model names from configured models
        return this.settings.models.map(m => m.name);
    }

    async listProviderModels(providerName: LLMProvider): Promise<string[]> {
        const provider = this.getProvider(providerName);
        if (!provider || !provider.listModels) {
            return [];
        }
        return provider.listModels();
    }

    async healthCheck(): Promise<boolean> {
        if (!this.currentModel) {
            return false;
        }

        const provider = this.getProviderForModel(this.currentModel);
        return provider.healthCheck();
    }

    async checkProviderHealth(providerName: LLMProvider): Promise<boolean> {
        const provider = this.getProvider(providerName);
        if (!provider) {
            return false;
        }
        return provider.healthCheck();
    }

    getAvailableProviders(): LLMProvider[] {
        return Array.from(this.providers.keys());
    }

    getEnabledProviders(): ProviderConfig[] {
        return this.settings.providers.filter(p => p.enabled);
    }

    // Model management methods
    addModel(model: ModelConfig): void {
        // Check if model with same ID exists
        const existing = this.settings.models.findIndex(m => m.id === model.id);
        if (existing >= 0) {
            this.settings.models[existing] = model;
        } else {
            this.settings.models.push(model);
        }
    }

    removeModel(modelId: string): boolean {
        const index = this.settings.models.findIndex(m => m.id === modelId);
        if (index >= 0) {
            this.settings.models.splice(index, 1);
            if (this.currentModel?.id === modelId) {
                this.currentModel = this.settings.models[0] || null;
            }
            return true;
        }
        return false;
    }

    updateModel(modelId: string, updates: Partial<ModelConfig>): boolean {
        const model = this.settings.models.find(m => m.id === modelId);
        if (model) {
            Object.assign(model, updates);
            return true;
        }
        return false;
    }

    getModels(): ModelConfig[] {
        return this.settings.models;
    }

    getModelsByProvider(provider: LLMProvider): ModelConfig[] {
        return this.settings.models.filter(m => m.provider === provider);
    }
}

// Singleton instance
let clientInstance: LLMClient | null = null;

export function getLLMClient(settings?: LLMWikiSettings): LLMClient {
    if (!clientInstance && settings) {
        clientInstance = new LLMClient(settings);
    } else if (clientInstance && settings) {
        clientInstance.updateSettings(settings);
    }
    if (!clientInstance) {
        throw new Error('LLMClient not initialized. Call getLLMClient with settings first.');
    }
    return clientInstance;
}

export function resetLLMClient(): void {
    clientInstance = null;
}