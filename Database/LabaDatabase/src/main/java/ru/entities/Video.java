package ru.entities;

import javax.persistence.*;
import java.util.List;
import java.util.UUID;

@Table(name = "VIDEO")
@Entity(name = "Video")
public class Video {
    private static final long serialVersionUID = -2583751117417450080L;

    @Id
    private UUID id;

    @Column(name = "name")
    private String name;

    @Column(name = "status")
    private String status;

    @OneToMany
    private List<Video> videos;

    @OneToOne
    @JoinColumn(name = "fileDescriptorId")
    private FileDescriptor fileDescriptor;

    @ManyToOne
    @JoinColumn(name = "cameraId")
    private Camera camera;

    public Video(){
        this.id = UUID.randomUUID();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public List<Video> getVideos() {
        return videos;
    }

    public void setVideos(List<Video> videos) {
        this.videos = videos;
    }

    public FileDescriptor getFileDescriptor() {
        return fileDescriptor;
    }

    public void setFileDescriptor(FileDescriptor fileDescriptor) {
        this.fileDescriptor = fileDescriptor;
    }

    public Camera getCamera() {
        return camera;
    }

    public void setCamera(Camera camera) {
        this.camera = camera;
    }
}