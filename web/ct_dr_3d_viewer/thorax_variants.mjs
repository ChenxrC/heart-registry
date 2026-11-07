const VALID_OVERRIDES = new Set(["auto", "registration", "display"]);
const CONCRETE_VARIANTS = new Set(["registration", "display"]);


export function resolveThoraxVariant(renderMode, override = "auto") {
  if (!VALID_OVERRIDES.has(override)) {
    throw new Error(`Unknown thorax variant: ${override}`);
  }
  if (override !== "auto") return override;
  return renderMode === "realistic" ? "display" : "registration";
}


export function resolveThoraxAssetRecords(thoraxAsset) {
  if (!thoraxAsset || typeof thoraxAsset !== "object") {
    throw new TypeError("Thorax asset record must be an object");
  }

  const registration = thoraxAsset.registration ?? thoraxAsset;
  const display = thoraxAsset.display ?? registration;
  if (typeof registration.path !== "string" || registration.path.length === 0) {
    throw new Error("Thorax registration asset is missing its path");
  }
  if (typeof display.path !== "string" || display.path.length === 0) {
    throw new Error("Thorax display asset is missing its path");
  }
  return Object.freeze({
    registration,
    display,
    hasDistinctDisplay: display !== registration && display.path !== registration.path,
  });
}


function variantGroups(entry) {
  if (Array.isArray(entry)) return entry;
  if (entry && typeof entry === "object" && "visible" in entry) return [entry];
  if (entry && typeof entry === "object") return Object.values(entry);
  return [];
}


export function setThoraxVariantVisibility(groups, variant) {
  if (!CONCRETE_VARIANTS.has(variant)) {
    throw new Error(`Thorax visibility requires a concrete variant, got: ${variant}`);
  }
  for (const [name, entry] of Object.entries(groups)) {
    for (const group of variantGroups(entry)) {
      if (!group || typeof group !== "object" || !("visible" in group)) {
        throw new TypeError(`Thorax ${name} group does not expose visibility`);
      }
      group.visible = name === variant;
    }
  }
  return variant;
}
