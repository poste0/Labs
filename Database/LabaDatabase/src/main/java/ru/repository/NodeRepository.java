package ru.repository;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;
import ru.entities.Node;

import java.util.UUID;

@Repository
public interface NodeRepository extends CrudRepository<Node, UUID> {
}
