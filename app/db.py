from typing import Dict, List, Optional

import aiomysql
from app.config import settings

_pool: Optional[aiomysql.Pool] = None


async def init_db() -> None:
    global _pool
    _pool = await aiomysql.create_pool(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        db=settings.mysql_database,
        autocommit=True,
        minsize=1,
        maxsize=5,
    )


async def close_db() -> None:
    global _pool
    if _pool:
        _pool.close()
        await _pool.wait_closed()
        _pool = None


async def fetch_products() -> List[Dict]:
    if _pool is None:
        return []
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT id, name, short_name, price, sale_price, in_stock, sku,"
                " COALESCE(NULLIF(sale_price, 0), NULLIF(price, 0), 0) AS final_price"
                " FROM product"
                " HAVING final_price > 0 ORDER BY name"
            )
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def search_products(query: str, limit: int = 10) -> List[Dict]:
    if _pool is None:
        return []
    pattern = f"%{query}%"
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT id, name, short_name, price, sale_price, in_stock, sku"
                " FROM product WHERE name LIKE %s OR short_name LIKE %s LIMIT %s",
                (pattern, pattern, limit),
            )
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def fetch_order(order_id: int) -> Optional[Dict]:
    if _pool is None:
        return None
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT id, status, first_name, last_name, address, city, postcode, phone"
                " FROM `order` WHERE id = %s",
                (order_id,),
            )
            row = await cur.fetchone()
    return dict(row) if row else None


async def fetch_order_products(order_id: int) -> List[Dict]:
    if _pool is None:
        return []
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT op.id, op.quantity, op.total,"
                " p.id AS product_id, p.name, p.short_name, p.sku, p.price, p.sale_price"
                " FROM order_product op"
                " JOIN product p ON op.product_id = p.id"
                " WHERE op.order_id = %s",
                (order_id,),
            )
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def update_order_address(
    order_id: int,
    first_name: str,
    last_name: str,
    address: str,
    city: str,
    postcode: str,
    phone: str,
) -> None:
    if _pool is None:
        return
    async with _pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE `order` SET first_name=%s, last_name=%s, address=%s,"
                " city=%s, postcode=%s, phone=%s WHERE id=%s",
                (first_name, last_name, address, city, postcode, phone, order_id),
            )


async def add_order_product(
    order_id: int, product_id: int, quantity: int, total: float
) -> None:
    if _pool is None:
        return
    async with _pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO order_product (order_id, product_id, quantity, total)"
                " VALUES (%s, %s, %s, %s)",
                (order_id, product_id, quantity, total),
            )


async def remove_order_product(order_product_id: int) -> None:
    if _pool is None:
        return
    async with _pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM order_product WHERE id = %s",
                (order_product_id,),
            )


async def get_product_by_id(product_id: int) -> Optional[Dict]:
    if _pool is None:
        return None
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT id, name, price FROM product WHERE id = %s",
                (product_id,),
            )
            row = await cur.fetchone()
    return dict(row) if row else None


async def get_products_by_ids(product_ids: List[int]) -> List[Dict]:
    if not product_ids or _pool is None:
        return []
    placeholders = ",".join(["%s"] * len(product_ids))
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                f"SELECT id, name, short_name, price, sale_price, in_stock, sku"
                f" FROM product WHERE id IN ({placeholders})",
                tuple(product_ids),
            )
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def fetch_all_orders() -> List[Dict]:
    if _pool is None:
        return []
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT id, status, first_name, last_name, email, total, date_created"
                " FROM `order` ORDER BY date_created DESC LIMIT 1000"
            )
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def fetch_low_stock(threshold: int = 5) -> List[Dict]:
    if _pool is None:
        return []
    async with _pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT id, name, short_name, price, sale_price, in_stock, sku"
                " FROM product WHERE in_stock <= %s ORDER BY in_stock ASC",
                (threshold,),
            )
            rows = await cur.fetchall()
    return [dict(row) for row in rows]
