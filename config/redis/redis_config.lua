-- Redis Lua脚本集合，用于分布式限流算法

-- 1. 令牌桶算法实现
-- KEYS[1]: 桶键名
-- KEYS[2]: 最后更新时间键名
-- ARGV[1]: 桶容量
-- ARGV[2]: 请求令牌数
-- ARGV[3]: 令牌生成速率（令牌/秒）
-- ARGV[4]: 当前时间戳（毫秒）
local token_bucket_script = [[
local bucket_key = KEYS[1]
local last_update_key = KEYS[2]
local capacity = tonumber(ARGV[1])
local tokens_requested = tonumber(ARGV[2])
local rate = tonumber(ARGV[3])
local current_time = tonumber(ARGV[4])

-- 获取当前令牌数和最后更新时间
local current_tokens = tonumber(redis.call('GET', bucket_key) or capacity)
local last_update = tonumber(redis.call('GET', last_update_key) or current_time)

-- 计算时间差
local time_diff = current_time - last_update

-- 计算应该生成的令牌数
local tokens_to_add = (time_diff / 1000) * rate

-- 更新当前令牌数（不能超过容量）
current_tokens = math.min(capacity, current_tokens + tokens_to_add)

-- 检查是否有足够令牌
if current_tokens >= tokens_requested then
    current_tokens = current_tokens - tokens_requested

    -- 更新Redis
    redis.call('SET', bucket_key, current_tokens)
    redis.call('SET', last_update_key, current_time)
    redis.call('EXPIRE', bucket_key, math.ceil(capacity / rate) + 1)
    redis.call('EXPIRE', last_update_key, math.ceil(capacity / rate) + 1)

    return {1, tostring(current_tokens)}  -- 1表示成功
else
    -- 令牌不足，更新状态但不扣减
    redis.call('SET', bucket_key, current_tokens)
    redis.call('SET', last_update_key, current_time)
    redis.call('EXPIRE', bucket_key, math.ceil(capacity / rate) + 1)
    redis.call('EXPIRE', last_update_key, math.ceil(capacity / rate) + 1)

    return {0, tostring(current_tokens)}  -- 0表示失败
end
]]

-- 2. 滑动窗口算法实现
-- KEYS[1]: 窗口键名
-- ARGV[1]: 窗口大小（秒）
-- ARGV[2]: 请求限制数
-- ARGV[3]: 当前时间戳
local sliding_window_script = [[
local window_key = KEYS[1]
local window_size = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local current_time = tonumber(ARGV[3])

-- 清理过期的请求记录
local min_score = current_time - (window_size * 1000)
redis.call('ZREMRANGEBYSCORE', window_key, 0, min_score)

-- 获取当前窗口内的请求数
local current_count = redis.call('ZCARD', window_key)

-- 检查是否超过限制
if current_count < limit then
    -- 添加当前请求
    redis.call('ZADD', window_key, current_time, current_time)
    redis.call('EXPIRE', window_key, window_size + 1)

    return {1, tostring(current_count + 1)}  -- 1表示成功
else
    return {0, tostring(current_count)}  -- 0表示失败
end
]]

-- 3. 固定窗口算法实现
-- KEYS[1]: 窗口键名
-- ARGV[1]: 窗口大小（秒）
-- ARGV[2]: 请求限制数
-- ARGV[3]: 当前时间戳
local fixed_window_script = [[
local window_key = KEYS[1]
local window_size = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local current_time = tonumber(ARGV[3])

local window_start = current_time - (current_time % window_size)
local current_window_key = window_key .. ':' .. window_start

-- 获取当前窗口计数
local current_count = tonumber(redis.call('GET', current_window_key) or 0)

-- 检查是否超过限制
if current_count < limit then
    current_count = current_count + 1
    redis.call('INCR', current_window_key)
    redis.call('EXPIRE', current_window_key, window_size)

    return {1, tostring(current_count)}  -- 1表示成功
else
    return {0, tostring(current_count)}  -- 0表示失败
end
]]

-- 4. 多维度限流脚本（支持多种限流规则）
-- KEYS[1]: 请求唯一标识
-- ARGV[1]: 限流规则JSON字符串
-- ARGV[2]: 当前时间戳
local multi_dimension_rate_limit_script = [[
local request_id = KEYS[1]
local rules_json = ARGV[1]
local current_time = tonumber(ARGV[2])

-- 解析限流规则
local rules = cjson.decode(rules_json)
local results = {}

for i, rule in ipairs(rules) do
    local rule_key = rule.key
    local rule_type = rule.type
    local limit = rule.limit
    local window = rule.window

    if rule_type == "token_bucket" then
        -- 令牌桶逻辑
        local bucket_key = "tb:" .. rule_key
        local last_update_key = "tb:update:" .. rule_key
        local capacity = rule.capacity or limit
        local rate = rule.rate or (limit / window)

        local current_tokens = tonumber(redis.call('GET', bucket_key) or capacity)
        local last_update = tonumber(redis.call('GET', last_update_key) or current_time)

        local time_diff = current_time - last_update
        local tokens_to_add = (time_diff / 1000) * rate
        current_tokens = math.min(capacity, current_tokens + tokens_to_add)

        if current_tokens >= 1 then
            current_tokens = current_tokens - 1
            redis.call('SET', bucket_key, current_tokens)
            redis.call('SET', last_update_key, current_time)
            redis.call('EXPIRE', bucket_key, math.ceil(capacity / rate) + 1)
            redis.call('EXPIRE', last_update_key, math.ceil(capacity / rate) + 1)
            results[i] = {type = rule_type, allowed = true, remaining = math.floor(current_tokens)}
        else
            redis.call('SET', bucket_key, current_tokens)
            redis.call('SET', last_update_key, current_time)
            redis.call('EXPIRE', bucket_key, math.ceil(capacity / rate) + 1)
            redis.call('EXPIRE', last_update_key, math.ceil(capacity / rate) + 1)
            results[i] = {type = rule_type, allowed = false, remaining = 0}
        end

    elseif rule_type == "sliding_window" then
        -- 滑动窗口逻辑
        local window_key = "sw:" .. rule_key

        local min_score = current_time - (window * 1000)
        redis.call('ZREMRANGEBYSCORE', window_key, 0, min_score)

        local current_count = redis.call('ZCARD', window_key)

        if current_count < limit then
            redis.call('ZADD', window_key, current_time, request_id .. ":" .. i)
            redis.call('EXPIRE', window_key, window + 1)
            results[i] = {type = rule_type, allowed = true, remaining = limit - current_count - 1}
        else
            results[i] = {type = rule_type, allowed = false, remaining = 0}
        end

    elseif rule_type == "fixed_window" then
        -- 固定窗口逻辑
        local window_start = current_time - (current_time % (window * 1000))
        local window_key = "fw:" .. rule_key .. ":" .. window_start

        local current_count = tonumber(redis.call('GET', window_key) or 0)

        if current_count < limit then
            redis.call('INCR', window_key)
            redis.call('EXPIRE', window_key, window)
            results[i] = {type = rule_type, allowed = true, remaining = limit - current_count - 1}
        else
            results[i] = {type = rule_type, allowed = false, remaining = 0}
        end
    end
end

-- 检查是否所有规则都通过
local all_allowed = true
for _, result in ipairs(results) do
    if not result.allowed then
        all_allowed = false
        break
    end
end

return cjson.encode({
    allowed = all_allowed,
    results = results,
    timestamp = current_time
})
]]

-- 5. 带权重的令牌桶算法
-- 支持不同请求消耗不同数量的令牌
local weighted_token_bucket_script = [[
local bucket_key = KEYS[1]
local last_update_key = KEYS[2]
local capacity = tonumber(ARGV[1])
local tokens_requested = tonumber(ARGV[2])
local rate = tonumber(ARGV[3])
local current_time = tonumber(ARGV[4])

local current_tokens = tonumber(redis.call('GET', bucket_key) or capacity)
local last_update = tonumber(redis.call('GET', last_update_key) or current_time)

local time_diff = current_time - last_update
local tokens_to_add = (time_diff / 1000) * rate

current_tokens = math.min(capacity, current_tokens + tokens_to_add)

if current_tokens >= tokens_requested then
    current_tokens = current_tokens - tokens_requested

    redis.call('SET', bucket_key, current_tokens)
    redis.call('SET', last_update_key, current_time)
    redis.call('EXPIRE', bucket_key, math.ceil(capacity / rate) + 1)
    redis.call('EXPIRE', last_update_key, math.ceil(capacity / rate) + 1)

    return {1, tostring(current_tokens)}
else
    redis.call('SET', bucket_key, current_tokens)
    redis.call('SET', last_update_key, current_time)
    redis.call('EXPIRE', bucket_key, math.ceil(capacity / rate) + 1)
    redis.call('EXPIRE', last_update_key, math.ceil(capacity / rate) + 1)

    return {0, tostring(current_tokens)}
end
]]

-- 6. 分布式计数器限流
-- 适用于简单的计数限流场景
local counter_rate_limit_script = [[
local counter_key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local current_time = tonumber(ARGV[3])

local current_count = tonumber(redis.call('GET', counter_key) or 0)

if current_count < limit then
    redis.call('INCR', counter_key)
    redis.call('EXPIRE', counter_key, window)

    return {1, tostring(current_count + 1)}
else
    return {0, tostring(current_count)}
end
]]

-- 7. 热点检测限流
-- 检测并限制高频访问的键
local hotspot_detection_script = [[
local hotspot_key = KEYS[1]
local base_key = KEYS[2]
local hotspot_threshold = tonumber(ARGV[1])
local hotspot_window = tonumber(ARGV[2])
local base_limit = tonumber(ARGV[3])
local base_window = tonumber(ARGV[4])
local current_time = tonumber(ARGV[5])

-- 检测热点
local min_score = current_time - (hotspot_window * 1000)
redis.call('ZREMRANGEBYSCORE', hotspot_key, 0, min_score)
local hotspot_count = redis.call('ZCARD', hotspot_key)

-- 记录当前访问
redis.call('ZADD', hotspot_key, current_time, current_time)
redis.call('EXPIRE', hotspot_key, hotspot_window + 1)

if hotspot_count > hotspot_threshold then
    -- 检测到热点，使用更严格的限流
    local strict_limit = math.floor(base_limit / 2)
    local strict_window = math.floor(base_window / 2)

    local strict_key = "strict:" .. base_key
    local strict_count = tonumber(redis.call('GET', strict_key) or 0)

    if strict_count < strict_limit then
        redis.call('INCR', strict_key)
        redis.call('EXPIRE', strict_key, strict_window)

        return {2, tostring(strict_count + 1), "hotspot_strict"}  -- 2表示热点严格模式
    else
        return {0, tostring(strict_count), "hotspot_blocked"}  -- 0表示热点拒绝
    end
else
    -- 正常限流
    local normal_count = tonumber(redis.call('GET', base_key) or 0)

    if normal_count < base_limit then
        redis.call('INCR', base_key)
        redis.call('EXPIRE', base_key, base_window)

        return {1, tostring(normal_count + 1), "normal"}  -- 1表示正常通过
    else
        return {0, tostring(normal_count), "normal_blocked"}  -- 0表示正常拒绝
    end
end
]]

-- 导出所有脚本
return {
    token_bucket = token_bucket_script,
    sliding_window = sliding_window_script,
    fixed_window = fixed_window_script,
    multi_dimension = multi_dimension_rate_limit_script,
    weighted_token_bucket = weighted_token_bucket_script,
    counter = counter_rate_limit_script,
    hotspot_detection = hotspot_detection_script
}