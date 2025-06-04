from ..clients import LoadStoreBytes, OpTypeCounts
from ..core.data import Launch


# Function to compute metrics in the analysis shown during visualization.
def analyze_records(records):
    data = []
    record_data = None
    for record in records:
        if isinstance(record, Launch):
            if record_data is not None:
                data.append(record_data)
            grid_size = record.grid_size
            record_data = [["Grid Size", tuple(grid_size)]]
        elif isinstance(record, OpTypeCounts):
            op_type_counts = record.type_counts
            record_data += [
                [op_type, count] for op_type, count in op_type_counts.items()
            ]
        elif isinstance(record, LoadStoreBytes):

            def calculate_ratio(record):
                return (
                    record.total_bytes_true / record.total_bytes_attempted
                    if record.total_bytes_attempted > 0
                    else 0
                )

            if record.type == "load":
                total_load_bytes_true = record.total_bytes_true
                overall_load_ratio = calculate_ratio(record)
                record_data.append(
                    ["Total number of bytes loaded", total_load_bytes_true]
                )
                record_data.append(["Masked Load Ratio", round(overall_load_ratio, 3)])
            elif record.type == "store":
                total_store_bytes_true = record.total_bytes_true
                overall_store_ratio = calculate_ratio(record)
                record_data.append(
                    ["Total number of bytes stored", total_store_bytes_true]
                )
                record_data.append(
                    ["Masked Store Ratio", round(overall_store_ratio, 3)]
                )
    if record_data is not None:
        data.append(record_data)

    return data
