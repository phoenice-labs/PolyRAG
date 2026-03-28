import { api } from './client'

export interface LLMProvider {
  id: string
  label: string
  protocol: string
  default_base_url: string
  requires_api_key: boolean
  notes: string
}

export interface LLMConfig {
  provider: string
  base_url: string
  api_key_set: boolean   // server never returns the actual key
  model: string
  temperature: number
  max_tokens: number
  timeout: number
}

export interface LLMConfigUpdate {
  provider: string
  base_url: string
  api_key: string        // sent to server; empty string = use provider default
  model: string
  temperature: number
  max_tokens: number
  timeout: number
}

export interface LLMTestResult {
  reachable: boolean
  provider: string
  base_url: string
  model: string
  error?: string
}

export const getLLMProviders = () =>
  api.get<LLMProvider[]>('/config/llm/providers').then((r) => r.data)

export const getLLMConfig = () =>
  api.get<LLMConfig>('/config/llm').then((r) => r.data)

export const updateLLMConfig = (cfg: LLMConfigUpdate) =>
  api.put<LLMConfig>('/config/llm', cfg).then((r) => r.data)

export const testLLMConnection = () =>
  api.get<LLMTestResult>('/config/llm/test').then((r) => r.data)
