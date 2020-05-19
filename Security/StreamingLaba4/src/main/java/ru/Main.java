package ru;

import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        FFmpegHelper ffmpeg = new FFmpegHelper();
        if (args.length < 2){
            throw new IllegalArgumentException("Need to enter file");
        }
        String fileName = args[0];
        String url = args[1];
        try {
            ffmpeg.getVideo(url, fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
