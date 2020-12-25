create table if not exists POCKET_JSON
(
  p_docs json unique
);
insert or
replace
into pocket_tab
select json_extract(P_DOCS, '$.item_id')                  item_id,
       json_extract(P_DOCS, '$.resolved_id')              resolved_id,
       json_extract(P_DOCS, '$.given_url')                given_url,
       json_extract(P_DOCS, '$.given_title')              given_title,
       json_extract(P_DOCS, '$.favorite')                 favorite,
       json_extract(P_DOCS, '$.status')                   status,
       json_extract(P_DOCS, '$.time_added')               time_added,
       json_extract(P_DOCS, '$.time_updated')             time_updated,
       json_extract(P_DOCS, '$.time_read')                time_read,
       json_extract(P_DOCS, '$.time_favorited')           time_favorited,
       json_extract(P_DOCS, '$.sort_id')                  sort_id,
       json_extract(P_DOCS, '$.resolved_title')           resolved_title,
       json_extract(P_DOCS, '$.resolved_url')             resolved_url,
       json_extract(P_DOCS, '$.excerpt')                  excerpt,
       json_extract(P_DOCS, '$.is_article')               is_article,
       json_extract(P_DOCS, '$.is_index')                 is_index,
       json_extract(P_DOCS, '$.has_video')                has_video,
       json_extract(P_DOCS, '$.has_image')                has_image,
       json_extract(P_DOCS, '$.word_count')               word_count,
       json_extract(P_DOCS, '$.lang')                     lang,
       json_extract(P_DOCS, '$.time_to_read')             time_to_read,
       json_extract(P_DOCS, '$.tags')                     tags,
       json_extract(P_DOCS, '$.listen_duration_estimate') listen_duration_estimate
from POCKET_JSON;
drop table if exists pocket_tags;
create table pocket_tags as
select distinct item_id, key
from pocket_tab,
     json_each(pocket_tab.tags);

select *
from POCKET_tab p1
  join pocket_tags p2
                 on p1.item_id = p2.item_id


SELECT *
  FROM POCKET_tab
 WHERE item_id IN (
           SELECT item_id
             FROM employees
            WHERE p1.tag = 'Canada'
       );
--------------------
--------------------
select *
from (
            select p1.*, p2.key tag
            from POCKET_tab p1
                        join pocket_tags p2
                             on p1.item_id = p2.item_id) p
where p.tag not in (
       select pt.key
       from pocket_tags pt
       where pt.item_id = p.item_id
);
drop table if exists pocket_missing_tags;
create table pocket_missing_tags as
select *
from (
       select item_id,
              group_concat(key, ',')
                new_tags
       from (
              select item_id,
                     given_title,
                     key,
                     coalesce(json_extract(tags, '$.' || key), 'valid', 'invalid') tag_valid_ind
              from pocket_tab p1
                     join
                   (
                     select distinct key
                     from pocket_tags
                     where key != 'r'
                       and key != 'py'
                   ) p2
                   on p1.given_title like '% ' || p2.key || ' %'
                     and p2.key not in (select key from pocket_tags where item_id = p1.item_id)
            )
       group by item_id
     );
