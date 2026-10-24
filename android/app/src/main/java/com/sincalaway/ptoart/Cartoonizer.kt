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

class NoFaceException : Exception("未检测到人脸，请使用清晰的正面单人照片")

/**
 * Android port of the ptoart pipeline:
 * face detection + eye-line alignment (ML Kit) -> upstream-style square crop ->
 * person mask (ML Kit selfie segmentation, replaces seg_model_384.pb) ->
 * photo2cartoon ONNX generator [1,3,256,256] in [-1,1].
 */
class Cartoonizer(modelFile: File) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val outputName: String

    private val faceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .build()
    )
    private val segmenter = Segmentation.getClient(
        SelfieSegmenterOptions.Builder()
            .setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE)
            .enableRawSizeMask()
            .build()
    )

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(max(1, Runtime.getRuntime().availableProcessors() / 2))
        }
        session = env.createSession(modelFile.absolutePath, opts)
        require(session.inputNames.size == 1) { "模型应只有一个输入" }
        inputName = session.inputNames.first()
        outputName = session.outputNames.firstOrNull { it == "output" } ?: session.outputNames.first()
    }

    /** @param background null = transparent, otherwise an ARGB color. */
    suspend fun convert(photo: Bitmap, strength: Float, background: Int?): Bitmap {
        val face = cropFace(photo)
        val mask = segment(face)
        return runGenerator(face, mask, strength.coerceIn(0f, 1f), background)
    }

    private suspend fun cropFace(src: Bitmap): Bitmap {
        val faces = faceDetector.process(InputImage.fromBitmap(src, 0)).await()
        val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
            ?: throw NoFaceException()

        // Align: rotate so the eye line is horizontal (upstream aligns with landmarks).
        val eyes = listOfNotNull(
            face.getLandmark(FaceLandmark.LEFT_EYE)?.position,
            face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
        ).sortedBy { it.x }
        val angle = if (eyes.size == 2)
            Math.toDegrees(atan2((eyes[1].y - eyes[0].y).toDouble(), (eyes[1].x - eyes[0].x).toDouble())).toFloat()
        else 0f
        val cx = face.boundingBox.exactCenterX()
        val cy = face.boundingBox.exactCenterY()
        val m = Matrix().apply { postRotate(-angle, cx, cy) }
        val aligned = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
        Canvas(aligned).apply {
            drawColor(Color.WHITE)
            drawBitmap(src, m, Paint(Paint.FILTER_BITMAP_FLAG))
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
        val crop = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(crop).apply {
            drawColor(Color.WHITE)
            drawBitmap(aligned, -left.toFloat(), -top.toFloat(), null)
        }
        aligned.recycle()
        return crop
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

    private fun runGenerator(face: Bitmap, fullMask: FloatArray, strength: Float, background: Int?): Bitmap {
        val n = 256
        val small = Bitmap.createScaledBitmap(face, n, n, true)
        val px = IntArray(n * n).also { small.getPixels(it, 0, n, 0, 0, n, n) }
        val fw = face.width
        val mask = FloatArray(n * n) { i ->
            val x = (i % n) * fw / n; val y = (i / n) * fw / n
            fullMask[y * fw + x]
        }
        val input = FloatBuffer.allocate(3 * n * n)
        val plane = n * n
        for (i in 0 until plane) {
            val c = px[i]; val a = mask[i]
            val rgb = intArrayOf(Color.red(c), Color.green(c), Color.blue(c))
            for (ch in 0..2) input.put(ch * plane + i, (rgb[ch] * a + (1 - a) * 255f) / 127.5f - 1f)
        }
        val out = OnnxTensor.createTensor(env, input, longArrayOf(1, 3, n.toLong(), n.toLong())).use { t ->
            session.run(mapOf(inputName to t)).use { r ->
                @Suppress("UNCHECKED_CAST")
                val o = (r.get(outputName).get() as OnnxTensor).floatBuffer
                FloatArray(o.remaining()).also { o.get(it) }
            }
        }
        require(out.size == 3 * plane) { "模型输出尺寸异常: ${out.size}" }
        val result = IntArray(plane)
        for (i in 0 until plane) {
            val c = px[i]; val a = mask[i]
            val src = intArrayOf(Color.red(c), Color.green(c), Color.blue(c))
            val ch = IntArray(3) { k ->
                val cart = (out[k * plane + i] + 1f) * 127.5f
                val fg = cart * strength + src[k] * (1 - strength)
                val v = if (background == null) fg else {
                    val bgc = intArrayOf(Color.red(background), Color.green(background), Color.blue(background))[k]
                    fg * a + bgc * (1 - a)
                }
                v.roundToInt().coerceIn(0, 255)
            }
            val alpha = if (background == null) (a * 255).roundToInt().coerceIn(0, 255) else 255
            result[i] = Color.argb(alpha, ch[0], ch[1], ch[2])
        }
        small.recycle()
        return Bitmap.createBitmap(result, n, n, Bitmap.Config.ARGB_8888)
    }

    override fun close() {
        faceDetector.close(); segmenter.close(); session.close()
    }
}
