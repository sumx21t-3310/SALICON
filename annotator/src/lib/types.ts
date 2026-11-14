/**
 * 多重解像度（フォビエーション）表示のパラメーター。
 * 既定値は SALICON 論文（Jiang et al., CVPR 2015）の値。
 */
export interface FoveationParams {
  /** 画像の元解像度における 1 度あたりの画素数。論文は 640x480 画像で 7.5。 */
  pixelsPerDegree: number;
  /** 解像度が半分になる視角（度）。論文の α = 2.5。 */
  halfResolutionDeg: number;
  /** ガウシアンピラミッドの段数。論文は 6（A1〜A6）。 */
  levels: number;
  /** カーソルとして描く赤い円の半径（度）。論文は 2 度。0 で非表示。 */
  cursorRadiusDeg: number;
}

export const DEFAULT_FOVEATION: FoveationParams = {
  pixelsPerDegree: 7.5,
  halfResolutionDeg: 2.5,
  levels: 6,
  cursorRadiusDeg: 2,
};

/** マウス位置の 1 サンプル。座標は画像の元解像度の画素。 */
export interface GazeSample {
  /** 画像の提示開始からの経過時間（ms） */
  t: number;
  x: number;
  y: number;
}

export interface TrialQuality {
  /** 平均サンプリング周波数（Hz） */
  samplingRateHz: number;
  /** 論文の基準（12 Hz 未満は破棄）を満たすか */
  samplingRateOk: boolean;
  /** カーソルが画像外にあった時間（ms） */
  outsideMs: number;
  /** 提示中にタブが非表示になった、またはウィンドウのフォーカスが外れたか */
  lostFocus: boolean;
  /** 一度もマウスが動かなかったか */
  noMovement: boolean;
}

export interface TrialResult {
  imageId: string;
  imageSrc: string;
  /** 画像の元解像度 */
  imageWidth: number;
  imageHeight: number;
  /** 画面上の表示サイズ（CSS px） */
  displayWidth: number;
  displayHeight: number;
  /** 提示時間（ms） */
  durationMs: number;
  /** 提示開始時刻（ISO 8601） */
  startedAt: string;
  /** 途中で中断された場合 false */
  completed: boolean;
  samples: GazeSample[];
  quality: TrialQuality;
  foveation: FoveationParams;
  pointerType: string;
  devicePixelRatio: number;
  userAgent: string;
}

export interface AnnotationImage {
  /** 結果の照合に使う ID。データセットでは画像ファイル名の stem を推奨。 */
  id: string;
  /** 画像の URL。WebGL で読むため、別オリジンの場合は CORS ヘッダーが必要。 */
  src: string;
}

export interface SessionResult {
  participantId?: string;
  startedAt: string;
  finishedAt: string;
  trials: TrialResult[];
}
