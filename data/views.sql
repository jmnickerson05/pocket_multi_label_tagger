CREATE VIEW IF NOT EXISTS v_all_labels as
select key tags, count(key) tag_count
from json_each(i.tags), items i
group by 1
order by 2 desc;
CREATE VIEW IF NOT EXISTS v_tbl_meta as
    select * from (select type, name, tbl_name 
    from sqlite_master where type in ('table', 'view')) s 
    cross join pragma_table_info(s.name) p
    order by name, cid;
CREATE VIEW IF NOT EXISTS v_latest_predictions as
    select item_id,
           model_probabilities
/*group by 1
having created_at = max(created_at)*/
    from model_output;
CREATE VIEW IF NOT EXISTS v_new_tags as
    with predicted_tags as (
        select item_id,
               key as tag --*
        from json_each(m.model_probabilities), v_latest_predictions m
        where cast(value as float) > 0.10
        order by 1
    ),
         existing_tags as (select item_id, key tag
                           from json_each(i.tags), items i
                           where i.tags is not null),
         new_tags as (select *
                      from (select *
                            from predicted_tags
                                except
                            select *
                            from existing_tags))
    select *
    from new_tags;
CREATE VIEW IF NOT EXISTS v_get_feature_text as
    select item_id,
           excerpt,
           html_text,
           clean_text(excerpt || ' ' || html_text) as combined_text,
           date_added
    from (select new.item_id,
                 new.excerpt,
                 get_html_text(new.resolved_url) html_text,
                 current_timestamp as            date_added
          from items new
          where new.item_id not in (
              select text_features.item_id
              from text_features)
            and new.excerpt is not null
         );
CREATE VIEW IF NOT EXISTS v_top_websites as
    select url, count(*) cnt
    from (
             select substr(url, 0, pos) url
             from (
                      select instr(replace(replace(given_url, 'http://', ''), 'https://', ''), '/') pos,
                             replace(replace(given_url, 'http://', ''), 'https://', '')             url,
                             given_url                                                              original_url
                      from items
                  )
         )
    group by 1
    order by 2 desc;
CREATE VIEW IF NOT EXISTS v_tags_unnested as
    select *
    from items i,
         json_each(i.tags) j;
CREATE VIEW IF NOT EXISTS v_tob_websites_by_tag as
with toptags as (
    select t.key    tag,
           count(*) cnt
    from v_tags_unnested t
    where t.key is not null
    group by 1
    order by 2 desc
    limit 10
)
select url, tag, count(*) cnt
from (
         select substr(url, 0, pos) url, tag
         from (
                  select instr(replace(replace(given_url, 'http://', ''), 'https://', ''), '/') pos,
                         replace(replace(given_url, 'http://', ''), 'https://', '')             url,
                         given_url                                                              original_url,
                         j.key                                                                  tag
                  from items i,
                       json_each(i.tags) j
              )
     )
where tag in (
    select tag
    from toptags
)
group by 1, 2
order by 3 desc;
CREATE VIEW IF NOT EXISTS v_top_10_tags as
    select t.key    tag,
           count(*) cnt
    from v_tags_unnested t
    where tag is not null
    group by 1
    order by 2 desc
    limit 10;
CREATE VIEW IF NOT EXISTS v_cummulative_tag_cnts_by_month as
    with dates as (select strftime('%Y-%m', date) yr_mon
                   from (
                            WITH RECURSIVE dates(date) AS (
                                /* adapted from:
                                https://stackoverflow.com/questions/32982372/how-to-generate-all-dates-between-two-dates
                                */
                                select date(min(datetime(time_added, 'unixepoch', 'localtime')), 'start of month')
                                from v_tags_unnested
                                UNION ALL
                                SELECT date(date, '+1 month')
                                FROM dates
                                WHERE date <
                                      (select date(max(datetime(time_added, 'unixepoch', 'localtime')),
                                                   'start of month')
                                       from v_tags_unnested)
                            )
                            SELECT date
                            FROM dates
                        )) --select * from dates
            ,
         tag_dates as (select yr_mon, tag
                       from v_top_10_tags t
                                cross join dates
         )
    select yr_mon,
           tag, -- cnt,
           sum(cnt) over (partition by tag
               order by tag, yr_mon asc rows between unbounded preceding and current row) cnt
    from (
             select tdates.*, coalesce(cnts.cnt, 0) cnt
             from tag_dates tdates
                      left join (
                 select t.key                                                               tag,
                        count(*)                                                            cnt,
                        strftime('%Y-%m', datetime(t.time_added, 'unixepoch', 'localtime')) date_added
--                 date(t.date_added) date_added
                 from v_tags_unnested t
                 where t.key in (select tag from v_top_10_tags)
                 group by 1, 3
                 order by 1, 3
             ) cnts
                                on tdates.tag = cnts.tag
                                    and tdates.yr_mon = cnts.date_added
             order by tdates.tag, tdates.yr_mon
         ) c