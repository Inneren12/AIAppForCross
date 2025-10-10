package com.appforcross.editor.model

enum class UnitMode { ST_PER_INCH, ST_PER_CM }
enum class SizePick { BY_WIDTH, BY_HEIGHT, BY_DPI }
enum class DenoiseLevel { NONE, LOW, MEDIUM, HIGH }
enum class DitheringType { NONE, FLOYD_STEINBERG, ATKINSON }
enum class ColorMetric { DE2000, DE76, OKLAB }
enum class ResamplingMode { AVERAGE_PER_CELL, SIMPLE_FAST }