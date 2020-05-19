package ru.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.util.FileCopyUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

import javax.servlet.http.HttpServletResponse;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

@Controller
public class MainController {
    @Value("${filePath}")
    private String filePath;

    @RequestMapping(value = "/video", method = RequestMethod.GET, produces = "application/vnd.apple.mpegurl")
    public @ResponseBody byte[] getVideo(HttpServletResponse response, @RequestParam("name") String name) throws IOException {
        response.addHeader("Content-Type", "application/vnd.apple.mpegurl");
        response.addHeader("Accept-Ranges", "bytes");
        response.addHeader("Access-Control-Allow-Origin", "*");
        response.addHeader("Access-Control-Expose-Headers", "Content-Length");

        File file = new File(filePath + name);
        byte[] bytes = FileCopyUtils.copyToByteArray(file);

        return bytes;
    }

    @RequestMapping(value = "/{name}", method = RequestMethod.GET, produces = "application/vnd.apple.mpegurl")
    public @ResponseBody byte[] getPiece(HttpServletResponse response, @PathVariable("name") String name) throws IOException {
        return getVideo(response, name);
    }
}
