package ru;

import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.FrameRecorder;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationContext;
import ru.service.FFMpegService;

import java.io.IOException;

@SpringBootApplication
public class Main {
    public static void main(String[] args) {


        FFMpegService service = new FFMpegService();
        try {
            service.getVideo("rtsp://rtsp_user:testrtsp@91.222.129.138:65113", "/home/sergei/Labs/Security/Laba4/stream.m3u8");
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        } catch (FrameRecorder.Exception e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        SpringApplication.run(Main.class);
    }
}
