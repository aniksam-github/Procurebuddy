package com.procurebuddy.repository;

import com.procurebuddy.entity.DocumentEntity;
import java.util.Optional;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface DocumentRepository extends JpaRepository<DocumentEntity, Long> {

    Optional<DocumentEntity> findByFileNameIgnoreCase(String fileName);

    @Query("""
            select d
            from DocumentEntity d
            where lower(d.fileName) like lower(concat('%', :query, '%'))
               or lower(d.content) like lower(concat('%', :query, '%'))
            order by d.updatedAt desc
            """)
    Page<DocumentEntity> search(@Param("query") String query, Pageable pageable);
}
