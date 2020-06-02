package ru.entities;

import javax.persistence.*;
import java.util.Date;
import java.util.UUID;

@Table(name = "FILE_DESCRIPTOR")
@Entity(name = "FileDescriptor")
public class FileDescriptor{
    private static final long serialVersionUID = -8246756918734123224L;

    @Id
    private UUID id;

    @Column(name = "name")
    private String name;

    @Column(name = "extension")
    private String extension;

    @Column(name = "size")
    private Double size;

    @Column(name = "createDate")
    private Date createDate;

    @OneToOne
    @JoinColumn(name = "video_id")
    private Video video;

    public FileDescriptor(){
        this.id = UUID.randomUUID();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getExtension() {
        return extension;
    }

    public void setExtension(String extension) {
        this.extension = extension;
    }

    public Double getSize() {
        return size;
    }

    public void setSize(Double size) {
        this.size = size;
    }

    public Date getCreateDate() {
        return createDate;
    }

    public void setCreateDate(Date createDate) {
        this.createDate = createDate;
    }

    public Video getVideo() {
        return video;
    }

    public void setVideo(Video video) {
        this.video = video;
    }
}