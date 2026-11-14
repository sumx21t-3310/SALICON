export { SaliconAnnotator } from "./SaliconAnnotator";
export type { SaliconAnnotatorProps } from "./SaliconAnnotator";
export { SaliconSession } from "./SaliconSession";
export type { SaliconSessionProps } from "./SaliconSession";
export { FoveatedRenderer, loadImage } from "./foveation";
export {
  resample,
  removeSaccades,
  isValidTrial,
  extractFixations,
  gazeTrack,
  buildFixationCounts,
  buildSaliencyMap,
  gaussianBlur,
} from "./processing";
export type { ProcessOptions, MapOptions } from "./processing";
export { DEFAULT_FOVEATION } from "./types";
export type {
  FoveationParams,
  GazeSample,
  TrialQuality,
  TrialResult,
  AnnotationImage,
  SessionResult,
} from "./types";
