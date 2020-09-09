package ru.controller;

import com.sun.org.apache.xalan.internal.xsltc.util.IntegerArray;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.util.FileCopyUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.nio.file.Files;

@Controller
public class MainController {
    @Value("${filePath}")
    private String filePath;

    @RequestMapping(value = "/video", method = RequestMethod.GET, produces = "application/vnd.apple.mpegurl")
    public @ResponseBody void getVideo(HttpServletRequest request, HttpServletResponse response, @RequestParam("name") String name) throws IOException {
        File file = new File(filePath + name);
        try {
            MultipartFileSender.fromPath(file.toPath()).with(request).with(response).serveResource();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @RequestMapping(value = "/{name}", method = RequestMethod.GET, produces = "application/vnd.apple.mpegurl")
    public @ResponseBody void getPiece(HttpServletRequest request, HttpServletResponse response, @PathVariable("name") String name) throws IOException {
        getVideo(request, response, name);
    }
}
