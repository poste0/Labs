package ru.repository;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;
import ru.entities.Video;

import java.util.List;
import java.util.UUID;

@Repository
public interface VideoRepository extends CrudRepository<Video, UUID> {
    List<Video> findAll();
}
