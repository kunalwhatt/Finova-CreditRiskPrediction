-- FACT TABLE
CREATE TABLE loan_fact (
    loan_id INT PRIMARY KEY,
    applicant_id INT,
    time_id INT,
    loan_amount FLOAT,
    emi FLOAT,
    default_flag INT,
    pd_score FLOAT
);

-- DIMENSIONS
CREATE TABLE applicant_dim (
    applicant_id INT PRIMARY KEY,
    age INT,
    income FLOAT,
    employment_type VARCHAR(50)
);

CREATE TABLE credit_dim (
    applicant_id INT,
    credit_history_months INT,
    active_loans INT,
    past_delinquency INT
);

CREATE TABLE device_dim (
    applicant_id INT,
    new_device_flag INT,
    kyc_mismatch INT
);

CREATE TABLE time_dim (
    time_id INT PRIMARY KEY,
    day INT,
    month INT,
    year INT
);
