package ru;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.context.ApplicationContext;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Service;

import java.io.IOException;

@Service
public class StreamService {
    @Autowired
    private Environment environment;

    private static final String url = "";

    public void startStream(){
        String filePath = environment.getProperty("filePath");
        System.out.println(filePath);

        try {
            Runtime.getRuntime().exec("java -jar streamer.jar " + filePath + "stream.m3u8 " + url);
        } catch (IOException e) {

            e.printStackTrace();
        }
    }
}
