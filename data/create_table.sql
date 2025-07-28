-- Skrypt SQL do utworzenia nowej tabeli products
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    category VARCHAR(255),
    name VARCHAR(255),
    description TEXT,
    image_path VARCHAR(255)
);

-- Sprawdź czy tabela została utworzona
\d products;
