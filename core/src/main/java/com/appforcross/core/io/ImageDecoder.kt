package com.appforcross.core.io

import com.appforcross.core.image.DecodedImage

interface ImageDecoder {
        fun decode(bytes: ByteArray): DecodedImage
    }