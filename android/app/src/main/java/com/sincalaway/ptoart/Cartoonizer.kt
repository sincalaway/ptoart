package com.sincalaway.ptoart

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import kotlinx.coroutines.tasks.await
import java.io.Closeable
import java.io.File
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class NoFaceException : Exception("未检测到人脸，请使用清晰的正面照片")

/** Which kind of ONNX file the user imported. */
enum class ModelKind { AVATAR, STYLE, UPSCALER }

object ModelInspector {
    /** The avatar generator has a fixed [1,3,256,256] input; AnimeGANv2 styles have dynamic H and W. */
    fun kindOf(file: File): ModelKind {
        val env = OrtEnvironment.getEnvironment()
        env.createSession(file.absolutePath, OrtSession.SessionOptions()).use { s ->
            require(s.inputNames.size == 1) { "模型应只有一个输入" }
            val info = s.inputInfo.values.first().info as ai.onnxruntime.TensorInfo
            val shape = info.shape
            require(shape.size == 4 && shape[1] == 3L) { "不支持的模型输入形状" }
            if (s.metadata.customMetadata["ptoart_kind"] == "upscaler") return ModelKind.UPSCALER
            return if (shape[2] == 256L && shape[3] == 256L) ModelKind.AVATAR else ModelKind.STYLE
        }
    }
}

private fun sessionOptions() = OrtSession.SessionOptions().apply {
    setIntraOpNumThreads(max(1, Runtime.getRuntime().availableProcessors() / 2))
}

/**
 * Real-ESRGAN compact x4 upscaler (tools/export_upscalers.py): NCHW RGB in [0,1], 4x output.
 */
class Upscaler(modelFile: File) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(modelFile.absolutePath, sessionOptions())
    private val inputName = session.inputNames.first()
    private val outputName = session.outputNames.first()
    val scale = session.metadata.customMetadata["ptoart_scale"]?.toIntOrNull() ?: 4

    /** Upscales a small bitmap (e.g. the 256px cartoon) in one pass. */
    fun upscale(src: Bitmap): Bitmap {
        val w = src.width; val h = src.height; val plane = w * h
        val px = IntArray(plane).also { src.getPixels(it, 0, w, 0, 0, w, h) }
        val input = FloatBuffer.allocate(3 * plane)
        for (i in 0 until plane) {
            val c = px[i]
            input.put(i, Color.red(c) / 255f)
            input.put(plane + i, Color.green(c) / 255f)
            input.put(2 * plane + i, Color.blue(c) / 255f)
        }
        val ow = w * scale; val oh = h * scale; val outPlane = ow * oh
        val out = OnnxTensor.createTensor(env, input, longArrayOf(1, 3, h.toLong(), w.toLong())).use { t ->
            session.run(mapOf(inputName to t)).use { r ->
                val o = (r.get(outputName).get() as OnnxTensor).floatBuffer
                FloatArray(o.remaining()).also { o.get(it) }
            }
        }
        require(out.size == 3 * outPlane) { "超分模型输出尺寸异常: ${out.size}" }
        val result = IntArray(outPlane) { i ->
            fun v(k: Int) = (out[k * outPlane + i] * 255f).roundToInt().coerceIn(0, 255)
            Color.rgb(v(0), v(1), v(2))
        }
        return Bitmap.createBitmap(result, ow, oh, Bitmap.Config.ARGB_8888)
    }

    override fun close() = session.close()
}

/**
 * Whole-image style generator: NCHW RGB in [-1,1], any size. Sides are rounded to a
 * multiple of 8, or of the model's `ptoart_multiple` metadata (16 for DCT-Net).
 */
class StyleModel(modelFile: File) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(modelFile.absolutePath, sessionOptions())
    private val inputName = session.inputNames.first()
    private val outputName = session.outputNames.first()
    private val multiple = max(8, session.metadata.customMetadata["ptoart_multiple"]?.toIntOrNull() ?: 8)

    fun stylize(photo: Bitmap, strength: Float, maxSide: Int = 1024): Bitmap {
        val scale = min(1f, maxSide.toFloat() / max(photo.width, photo.height))
        val w = max(multiple, (photo.width * scale / multiple).roundToInt() * multiple)
        val h = max(multiple, (photo.height * scale / multiple).roundToInt() * multiple)
        val small = Bitmap.createScaledBitmap(photo, w, h, true)
        val plane = w * h
        val px = IntArray(plane).also { small.getPixels(it, 0, w, 0, 0, w, h) }
        val input = FloatBuffer.allocate(3 * plane)
        for (i in 0 until plane) {
            val c = px[i]
            input.put(i, Color.red(c) / 127.5f - 1f)
            input.put(plane + i, Color.green(c) / 127.5f - 1f)
            input.put(2 * plane + i, Color.blue(c) / 127.5f - 1f)
        }
        val out = OnnxTensor.createTensor(env, input, longArrayOf(1, 3, h.toLong(), w.toLong())).use { t ->
            session.run(mapOf(inputName to t)).use { r ->
                val o = (r.get(outputName).get() as OnnxTensor).floatBuffer
                FloatArray(o.remaining()).also { o.get(it) }
            }
        }
        require(out.size == 3 * plane) { "风格模型输出尺寸异常: ${out.size}" }
        val s = strength.coerceIn(0f, 1f)
        val result = IntArray(plane) { i ->
            val c = px[i]
            fun ch(k: Int, src: Int) = ((out[k * plane + i] + 1f) * 127.5f * s + src * (1 - s))
                .roundToInt().coerceIn(0, 255)
            Color.rgb(ch(0, Color.red(c)), ch(1, Color.green(c)), ch(2, Color.blue(c)))
        }
        if (small !== photo) small.recycle()
        val styled = Bitmap.createBitmap(result, w, h, Bitmap.Config.ARGB_8888)
        if (w == photo.width && h == photo.height) return styled
        return Bitmap.createScaledBitmap(styled, photo.width, photo.height, true).also { styled.recycle() }
    }

    override fun close() = session.close()
}

/**
 * Android port of the ptoart face pipeline:
 * face detection + eye-line alignment (ML Kit) -> upstream-style square crop ->
 * person mask (ML Kit selfie segmentation, replaces seg_model_384.pb) ->
 * ptoart avatar ONNX generator [1,3,256,256] in [-1,1].
 */
class Cartoonizer(modelFile: File, private val painter: Boolean = false) : Closeable {
    /**
     * Generator side: 256 for the ptoart avatar model; for a face painter its
     * `ptoart_face_side` metadata (288 for DCT-Net heads), else 512 (AnimeGAN).
     */
    private val n: Int

    /** Optional super-resolution for enlarging the 256px cartoon; owned by the caller. */
    var upscaler: Upscaler? = null

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val outputName: String

    private val faceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .setMinFaceSize(0.05f)
            .build()
    )
    private val segmenter = Segmentation.getClient(
        SelfieSegmenterOptions.Builder()
            .setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE)
            .enableRawSizeMask()
            .build()
    )

    /** An aligned square face crop and the transform from photo pixels to crop pixels. */
    private class FaceCrop(val bitmap: Bitmap, val toCrop: Matrix)

    init {
        session = env.createSession(modelFile.absolutePath, sessionOptions())
        require(session.inputNames.size == 1) { "模型应只有一个输入" }
        inputName = session.inputNames.first()
        outputName = session.outputNames.firstOrNull { it == "output" } ?: session.outputNames.first()
        n = if (painter) session.metadata.customMetadata["ptoart_face_side"]?.toIntOrNull() ?: 512 else 256
    }

    /**
     * Square cartoon avatar of the largest face.
     * @param background null = transparent, otherwise an ARGB color.
     */
    suspend fun convertFace(photo: Bitmap, strength: Float, background: Int?, size: Int = 512): Bitmap {
        val crop = cropFaces(photo, 1).firstOrNull() ?: throw NoFaceException()
        val mask = segment(crop.bitmap)
        val layer = cartoonLayer(crop.bitmap, mask, strength.coerceIn(0f, 1f), size, feather = false)
        crop.bitmap.recycle()
        if (background == null) return layer
        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(out).apply {
            drawColor(background)
            drawBitmap(layer, 0f, 0f, null)
        }
        layer.recycle()
        return out
    }

    /** The whole photo with up to [maxFaces] cartoon faces blended back in place. */
    suspend fun convertPhoto(photo: Bitmap, strength: Float, maxFaces: Int = 10): Bitmap {
        val crops = cropFaces(photo, maxFaces)
        if (crops.isEmpty()) throw NoFaceException()
        val out = photo.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        // Paint smaller faces first so the largest face wins any overlap.
        for (crop in crops.asReversed()) {
            val side = crop.bitmap.width
            val mask = segment(crop.bitmap)
            val layer = cartoonLayer(crop.bitmap, mask, strength.coerceIn(0f, 1f), side, feather = true)
            val toPhoto = Matrix().also { crop.toCrop.invert(it) }
            canvas.drawBitmap(layer, toPhoto, paint)
            layer.recycle(); crop.bitmap.recycle()
        }
        return out
    }

    private suspend fun cropFaces(src: Bitmap, maxFaces: Int): List<FaceCrop> {
        val faces = faceDetector.process(InputImage.fromBitmap(src, 0)).await()
            .filter { it.boundingBox.width() >= 24 }
            .sortedByDescending { it.boundingBox.width() * it.boundingBox.height() }
            .take(maxFaces)
        return faces.map { cropFace(src, it) }
    }

    private fun cropFace(src: Bitmap, face: Face): FaceCrop {
        // Align: rotate so the eye line is horizontal (upstream aligns with landmarks).
        val eyes = listOfNotNull(
            face.getLandmark(FaceLandmark.LEFT_EYE)?.position,
            face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
        ).sortedBy { it.x }
        val angle = if (eyes.size == 2)
            Math.toDegrees(atan2((eyes[1].y - eyes[0].y).toDouble(), (eyes[1].x - eyes[0].x).toDouble())).toFloat()
        else 0f
        val m = Matrix().apply {
            postRotate(-angle, face.boundingBox.exactCenterX(), face.boundingBox.exactCenterY())
        }

        val pts = facePoints(face)
        val arr = FloatArray(pts.size * 2)
        pts.forEachIndexed { i, p -> arr[2 * i] = p.x; arr[2 * i + 1] = p.y }
        m.mapPoints(arr)
        var l = Float.MAX_VALUE; var r = -Float.MAX_VALUE
        var t = Float.MAX_VALUE; var b = -Float.MAX_VALUE
        for (i in pts.indices) {
            l = min(l, arr[2 * i]); r = max(r, arr[2 * i])
            t = min(t, arr[2 * i + 1]); b = max(b, arr[2 * i + 1])
        }
        // Same expansion as utils/preprocess.py (face oval starts above the brows,
        // so the upward expansion is a little smaller than upstream's 0.8).
        val h = b - t; val w = r - l
        var top = (t - 0.55f * h).roundToInt()
        var bottom = (b + 0.3f * h).roundToInt()
        var left = (l - 0.3f * w).roundToInt()
        var right = (r + 0.3f * w).roundToInt()
        if (bottom - top > right - left) {
            left -= ((bottom - top) - (right - left)) / 2; right = left + (bottom - top)
        } else {
            top -= ((right - left) - (bottom - top)) / 2; bottom = top + (right - left)
        }
        val size = bottom - top + 1
        m.postTranslate(-left.toFloat(), -top.toFloat())
        val crop = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(crop).apply {
            drawColor(Color.WHITE)
            drawBitmap(src, m, Paint(Paint.FILTER_BITMAP_FLAG))
        }
        return FaceCrop(crop, m)
    }

    private fun facePoints(face: Face): List<PointF> {
        val contour = listOf(FaceContour.FACE, FaceContour.LEFT_EYEBROW_TOP, FaceContour.RIGHT_EYEBROW_TOP)
            .flatMap { face.getContour(it)?.points ?: emptyList() }
        if (contour.isNotEmpty()) return contour
        val bb = face.boundingBox
        return listOf(PointF(bb.left.toFloat(), bb.top.toFloat()), PointF(bb.right.toFloat(), bb.bottom.toFloat()))
    }

    /** Returns a size*size alpha mask in [0,1]. */
    private suspend fun segment(face: Bitmap): FloatArray {
        val res = segmenter.process(InputImage.fromBitmap(face, 0)).await()
        val buf = res.buffer.order(ByteOrder.nativeOrder()).asFloatBuffer()
        val mw = res.width; val mh = res.height
        val raw = FloatArray(mw * mh).also { buf.get(it) }
        if (mw == face.width && mh == face.height) return raw
        val out = FloatArray(face.width * face.height)
        for (y in 0 until face.height) for (x in 0 until face.width) {
            out[y * face.width + x] = raw[(y * mh / face.height) * mw + (x * mw / face.width)]
        }
        return out
    }

    /** Runs the generator at n x n and returns the cartoon in [-1,1] CHW order. */
    private fun infer(px: IntArray, mask: FloatArray, n: Int): FloatArray {
        val plane = n * n
        val input = FloatBuffer.allocate(3 * plane)
        for (i in 0 until plane) {
            // The face painter sees the raw crop; the avatar model expects a white background.
            val c = px[i]; val a = if (painter) 1f else mask[i]
            input.put(i, (Color.red(c) * a + (1 - a) * 255f) / 127.5f - 1f)
            input.put(plane + i, (Color.green(c) * a + (1 - a) * 255f) / 127.5f - 1f)
            input.put(2 * plane + i, (Color.blue(c) * a + (1 - a) * 255f) / 127.5f - 1f)
        }
        val out = OnnxTensor.createTensor(env, input, longArrayOf(1, 3, n.toLong(), n.toLong())).use { t ->
            session.run(mapOf(inputName to t)).use { r ->
                val o = (r.get(outputName).get() as OnnxTensor).floatBuffer
                FloatArray(o.remaining()).also { o.get(it) }
            }
        }
        require(out.size == 3 * plane) { "模型输出尺寸异常: ${out.size}" }
        return out
    }

    /**
     * Cartoon foreground of a face crop as an ARGB layer of [outSize]^2 whose alpha is the
     * person mask. The 256px cartoon is upscaled; alpha and the strength blend use the
     * full-resolution crop so edges stay sharp. [feather] softens edges for pasting.
     */
    private fun cartoonLayer(face: Bitmap, fullMask: FloatArray, strength: Float, outSize: Int, feather: Boolean): Bitmap {
        val fw = face.width
        val small = Bitmap.createScaledBitmap(face, n, n, true)
        val px = IntArray(n * n).also { small.getPixels(it, 0, n, 0, 0, n, n) }
        small.recycle()
        val maskN = FloatArray(n * n) { i -> fullMask[((i / n) * fw / n) * fw + (i % n) * fw / n] }
        val out = infer(px, maskN, n)
        val plane = n * n
        val cartoonPx = IntArray(plane) { i ->
            fun v(k: Int) = ((out[k * plane + i] + 1f) * 127.5f).roundToInt().coerceIn(0, 255)
            Color.rgb(v(0), v(1), v(2))
        }
        val cartoonN = Bitmap.createBitmap(cartoonPx, n, n, Bitmap.Config.ARGB_8888)
        val sr = upscaler
        val base = if (sr != null && outSize > n) sr.upscale(cartoonN).also { cartoonN.recycle() } else cartoonN
        val cartoon = Bitmap.createScaledBitmap(base, outSize, outSize, true)
        if (cartoon !== base) base.recycle()
        val source = Bitmap.createScaledBitmap(face, outSize, outSize, true)

        val total = outSize * outSize
        val cp = IntArray(total).also { cartoon.getPixels(it, 0, outSize, 0, 0, outSize, outSize) }
        val sp = IntArray(total).also { source.getPixels(it, 0, outSize, 0, 0, outSize, outSize) }
        cartoon.recycle(); if (source !== face) source.recycle()
        var alpha = FloatArray(total) { i -> fullMask[((i / outSize) * fw / outSize) * fw + (i % outSize) * fw / outSize] }
        if (feather) alpha = featherMask(alpha, outSize)
        val result = IntArray(total) { i ->
            val c = cp[i]; val s = sp[i]
            fun mix(a: Int, b: Int) = (a * strength + b * (1 - strength)).roundToInt().coerceIn(0, 255)
            Color.argb(
                (alpha[i] * 255).roundToInt().coerceIn(0, 255),
                mix(Color.red(c), Color.red(s)), mix(Color.green(c), Color.green(s)), mix(Color.blue(c), Color.blue(s))
            )
        }
        return Bitmap.createBitmap(result, outSize, outSize, Bitmap.Config.ARGB_8888)
    }

    /** Box-blur the mask and fade the crop border so pasted faces show no seams. */
    private fun featherMask(mask: FloatArray, n: Int): FloatArray {
        val radius = max(1, n / 128)
        val tmp = FloatArray(mask.size)
        val out = FloatArray(mask.size)
        for (y in 0 until n) {          // horizontal pass
            var sum = 0f
            for (x in -radius..radius) sum += mask[y * n + x.coerceIn(0, n - 1)]
            for (x in 0 until n) {
                tmp[y * n + x] = sum / (2 * radius + 1)
                sum += mask[y * n + (x + radius + 1).coerceIn(0, n - 1)] - mask[y * n + (x - radius).coerceIn(0, n - 1)]
            }
        }
        for (x in 0 until n) {          // vertical pass
            var sum = 0f
            for (y in -radius..radius) sum += tmp[y.coerceIn(0, n - 1) * n + x]
            for (y in 0 until n) {
                out[y * n + x] = sum / (2 * radius + 1)
                sum += tmp[(y + radius + 1).coerceIn(0, n - 1) * n + x] - tmp[(y - radius).coerceIn(0, n - 1) * n + x]
            }
        }
        val fade = max(1, (n * 0.12f).toInt())
        fun ramp(i: Int) = min(1f, min(i, n - 1 - i).toFloat() / fade)
        for (y in 0 until n) for (x in 0 until n) out[y * n + x] *= min(ramp(x), ramp(y))
        return out
    }

    override fun close() {
        faceDetector.close(); segmenter.close(); session.close()
    }
}
