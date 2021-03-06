package ru;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.IplImage;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static org.bytedeco.opencv.global.opencv_imgproc.cvSmooth;
import static org.opencv.imgproc.Imgproc.CV_GAUSSIAN;

public class FFmpegHelper {
    public void getVideo(String url, String fileName) throws IOException {
        final FFmpegFrameGrabber grabber = FFmpegFrameGrabber.createDefault(url);
        File file = new File(fileName);
        if(!file.exists()){
            file.createNewFile();
        }
        setupGrabber(grabber);
        grabber.start();
        FFmpegFrameRecorder recorder = FFmpegFrameRecorder.createDefault(file, grabber.getImageWidth(), grabber.getImageHeight());
        setupRecorder(recorder, grabber);
        recorder.start();
        Executor executor = Executors.newSingleThreadExecutor();
        executor.execute(() -> {
            while(true){
                try {
                    Frame frame = grabber.grab();
                    OpenCVFrameConverter converter = new OpenCVFrameConverter.ToIplImage();
                    IplImage image = converter.convertToIplImage(frame);
                    cvSmooth(image, image, CV_GAUSSIAN, 9, 9, 2, 2);

                    recorder.record(converter.convert(image));
                } catch (FrameGrabber.Exception e) {
                    e.printStackTrace();
                } catch (FrameRecorder.Exception e) {
                    e.printStackTrace();
                }
            }
        });

    }

    private FFmpegFrameRecorder setupRecorder(FFmpegFrameRecorder recorder, FFmpegFrameGrabber grabber){
        recorder.setVideoCodec(grabber.getVideoCodec());
        recorder.setVideoBitrate(grabber.getVideoBitrate());
        recorder.setFrameRate(grabber.getFrameRate());
        recorder.setOption("crf", "20");
        recorder.setOption("movflags", "faststart");
        recorder.setOption("sc_threshold", "0");
        recorder.setOption("g", String.valueOf(grabber.getFrameRate()));
        recorder.setAudioCodec(grabber.getAudioCodec());
        recorder.setAudioChannels(grabber.getAudioChannels());
        recorder.setOption("f", "hls");
        recorder.setOption("hls_time", "4");
        recorder.setOption("hls_flags", "round_durations");
        recorder.setOption("hls_flags", "delete_segments");

        return recorder;
    }

    private FFmpegFrameGrabber setupGrabber(FFmpegFrameGrabber grabber){
        grabber.setOption("rtsp_transport", "tcp");

        return grabber;
    }
}
