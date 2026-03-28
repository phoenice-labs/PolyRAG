/**
 * Collection name scoping — mirrors api/deps.py _model_slug / build_pipeline_config.
 *
 * When a collection is ingested, the backend appends a model slug so that
 * collections built with different embedding models never share the same
 * namespace (vectors have different dimensionality and are incompatible).
 *
 *   polyrag_docs  +  all-MiniLM-L6-v2       →  polyrag_docs_minilm
 *   polyrag_docs  +  BAAI/bge-base-en-v1.5  →  polyrag_docs_bge-base
 *   polyrag_docs  +  BAAI/bge-large-en-v1.5 →  polyrag_docs_bge-large
 */

export const MODEL_SLUGS: Record<string, string> = {
  'all-MiniLM-L6-v2':       'minilm',
  'BAAI/bge-base-en-v1.5':  'bge-base',
  'BAAI/bge-large-en-v1.5': 'bge-large',
}

/** Reverse map: slug → model id */
export const SLUG_TO_MODEL: Record<string, string> = Object.fromEntries(
  Object.entries(MODEL_SLUGS).map(([model, slug]) => [slug, model])
)

/** Return the slug for a given embedding model id (e.g. "minilm"). */
export function modelSlug(embeddingModel: string): string {
  return MODEL_SLUGS[embeddingModel]
    ?? embeddingModel.split('/').pop()?.toLowerCase().replace(/[^a-z0-9]/g, '-')
    ?? 'unknown'
}

/** Return the scoped collection name for a base name + model, e.g. "polyrag_docs_minilm". */
export function scopedCollectionName(baseName: string, embeddingModel: string): string {
  const slug = modelSlug(embeddingModel)
  return baseName.endsWith(`_${slug}`) ? baseName : `${baseName}_${slug}`
}

/** Extract the model slug suffix from a scoped collection name, or null if none. */
export function slugFromCollectionName(name: string): string | null {
  for (const slug of Object.values(MODEL_SLUGS)) {
    if (name.endsWith(`_${slug}`)) return slug
  }
  return null
}

/** Human-readable label: "polyrag_docs_minilm" → "polyrag_docs · MiniLM" */
export function collectionLabel(name: string): string {
  const slug = slugFromCollectionName(name)
  if (!slug) return name
  const modelShort =
    slug === 'minilm'    ? 'MiniLM' :
    slug === 'bge-base'  ? 'BGE-base' :
    slug === 'bge-large' ? 'BGE-large' :
    slug
  const base = name.slice(0, name.length - slug.length - 1)
  return `${base} · ${modelShort}`
}

/**
 * Given all known collection names and the current embedding model + active
 * collection, return the best candidate to pre-select:
 *  1. Exact match of activeCollection in known list
 *  2. activeCollection + current model slug (e.g. "polyrag_docs_minilm")
 *  3. Base-stripped + current model slug
 *  4. Any collection ending with the current model slug
 *  5. Alphabetically first (last resort)
 */
export function bestMatchCollection(
  known: string[],
  activeCollection: string,
  embeddingModel: string,
): string | null {
  if (known.length === 0) return null
  const slug = modelSlug(embeddingModel)

  if (known.includes(activeCollection)) return activeCollection

  const scoped = scopedCollectionName(activeCollection, embeddingModel)
  if (known.includes(scoped)) return scoped

  // strip any existing slug suffix then re-scope
  const base = activeCollection.replace(new RegExp(`_${Object.values(MODEL_SLUGS).join('|_')}$`), '')
  const rescoped = `${base}_${slug}`
  if (known.includes(rescoped)) return rescoped

  const slugMatch = known.find((n) => n.endsWith(`_${slug}`))
  if (slugMatch) return slugMatch

  return known[0]
}
